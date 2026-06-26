"""In-memory model registry.

Owns a single :class:`AdvancedMovieRecommender`, trains models on demand, caches
them for the life of the process, and exposes thread-safe accessors for the API
layer. This is the backend equivalent of the documentation's model/builder layer
— minus any database, since this service is intentionally stateless.

Training is guarded by a re-entrant lock so the startup warm-up thread and an
incoming ``POST /api/train`` request cannot corrupt shared state.
"""

from __future__ import annotations

import re
import threading
from enum import Enum

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from recommender import TENSORFLOW_AVAILABLE, AdvancedMovieRecommender
from recommender.data import MOVIES_COLUMNS

from app.utils.settings import settings

# Real MovieLens genre flag columns (everything after the metadata columns).
GENRE_COLUMNS = MOVIES_COLUMNS[5:]  # 'unknown', 'Action', ... 'Western'

ALGORITHMS = {
    "SVD": {
        "name": "SVD Matrix Factorization",
        "description": "Decomposes the user-item matrix using Singular Value Decomposition.",
    },
    "NMF": {
        "name": "Non-negative Matrix Factorization",
        "description": "Finds interpretable patterns in non-negative rating data.",
    },
    "Neural": {
        "name": "Neural Collaborative Filtering",
        "description": "Deep learning with user/item embeddings for complex interactions.",
    },
}


class Status(str, Enum):
    NOT_STARTED = "not_started"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"  # e.g. Neural CF without TensorFlow installed


class ModelRegistry:
    def __init__(self) -> None:
        self._rec = AdvancedMovieRecommender()
        self._lock = threading.RLock()

        self._data_loaded = False
        self._status: dict[str, Status] = {
            "SVD": Status.NOT_STARTED,
            "NMF": Status.NOT_STARTED,
            "Neural": Status.NOT_STARTED if TENSORFLOW_AVAILABLE else Status.UNAVAILABLE,
        }
        self._metrics: dict[str, dict] = {}
        self._errors: dict[str, str] = {}

        # Derived lookups, populated once data is loaded.
        self._avg_rating: dict[int, float] = {}
        self._user_ids: list[int] = []

    # ------------------------------------------------------------------ data
    def ensure_data(self) -> None:
        with self._lock:
            if self._data_loaded:
                return
            self._rec.load_data()
            self._rec.create_train_test_split()
            self._avg_rating = (
                self._rec.ratings.groupby("movie_id")["rating"].mean().round(2).to_dict()
            )
            self._user_ids = sorted(int(u) for u in self._rec.ratings["user_id"].unique())
            self._data_loaded = True

    # --------------------------------------------------------------- training
    def train(self, algorithm: str) -> dict:
        """Train one algorithm (idempotent-ish: retrains and refreshes metrics)."""
        if algorithm not in ALGORITHMS:
            raise KeyError(algorithm)
        if algorithm == "Neural" and not TENSORFLOW_AVAILABLE:
            self._status["Neural"] = Status.UNAVAILABLE
            raise RuntimeError("Neural CF unavailable: TensorFlow is not installed.")

        with self._lock:
            self.ensure_data()
            self._status[algorithm] = Status.TRAINING
            try:
                if algorithm == "SVD":
                    self._rec.build_svd_model(n_components=settings.SVD_COMPONENTS)
                    metrics = self._evaluate_mf(
                        self._rec.user_factors,
                        self._rec.movie_factors,
                        user_means=self._rec.user_means,
                    )
                elif algorithm == "NMF":
                    self._rec.build_nmf_model(n_components=settings.NMF_COMPONENTS)
                    metrics = self._evaluate_mf(
                        self._rec.user_factors_nmf,
                        self._rec.movie_factors_nmf,
                    )
                else:  # Neural
                    self._rec.build_neural_cf_model(
                        embedding_size=settings.NEURAL_EMBEDDING_SIZE
                    )
                    self._rec.train_neural_model(epochs=settings.NEURAL_EPOCHS)
                    metrics = self._evaluate_neural()

                self._metrics[algorithm] = metrics
                self._status[algorithm] = Status.READY
                self._errors.pop(algorithm, None)
                return metrics
            except Exception as exc:  # noqa: BLE001 - surface as status
                self._status[algorithm] = Status.FAILED
                self._errors[algorithm] = str(exc)
                raise

    def warm_up(self) -> None:
        """Train the fast models (SVD, NMF). Intended for a background thread."""
        for algorithm in ("SVD", "NMF"):
            try:
                self.train(algorithm)
            except Exception:  # noqa: BLE001 - status already records the failure
                pass

    # ------------------------------------------------------------ evaluation
    def _evaluate_mf(self, user_factors, movie_factors, user_means=None) -> dict:
        """Vectorized RMSE/MAE for a matrix-factorization model on the test set."""
        uim = self._rec.user_item_matrix
        user_pos = {u: i for i, u in enumerate(uim.index)}
        movie_pos = {m: j for j, m in enumerate(uim.columns)}

        test = self._rec.test_data
        mask = test["user_id"].isin(user_pos) & test["movie_id"].isin(movie_pos)
        t = test[mask]

        ui = t["user_id"].map(user_pos).to_numpy()
        mj = t["movie_id"].map(movie_pos).to_numpy()
        preds = np.sum(user_factors[ui] * movie_factors[mj], axis=1)
        if user_means is not None:
            preds = preds + user_means.to_numpy()[ui]

        actual = t["rating"].to_numpy()
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, preds))), 3),
            "mae": round(float(mean_absolute_error(actual, preds)), 3),
        }

    def _evaluate_neural(self) -> dict:
        rec = self._rec
        test = rec.test_data
        mask = test["user_id"].isin(rec.user_to_idx) & test["movie_id"].isin(rec.movie_to_idx)
        t = test[mask]
        ui = t["user_id"].map(rec.user_to_idx).to_numpy()
        mj = t["movie_id"].map(rec.movie_to_idx).to_numpy()
        preds = rec.neural_model.predict([ui, mj], verbose=0).flatten()
        actual = t["rating"].to_numpy()
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, preds))), 3),
            "mae": round(float(mean_absolute_error(actual, preds)), 3),
        }

    # -------------------------------------------------------------- accessors
    def status(self) -> dict:
        return {
            "data_loaded": self._data_loaded,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "models": {
                key: {
                    "status": self._status[key].value,
                    "metrics": self._metrics.get(key),
                    "error": self._errors.get(key),
                }
                for key in ALGORITHMS
            },
        }

    def algorithms(self) -> list[dict]:
        return [
            {
                "key": key,
                "name": meta["name"],
                "description": meta["description"],
                "status": self._status[key].value,
                "metrics": self._metrics.get(key),
            }
            for key, meta in ALGORITHMS.items()
        ]

    def users(self) -> list[int]:
        self.ensure_data()
        return self._user_ids

    def movie_info(self, movie_id: int) -> dict:
        row = self._rec.movies[self._rec.movies["movie_id"] == movie_id].iloc[0]
        title = str(row["title"])
        year_match = re.search(r"\((\d{4})\)", title)
        genres = [g for g in GENRE_COLUMNS if g != "unknown" and row[g] == 1]
        return {
            "movie_id": int(movie_id),
            "title": title,
            "year": int(year_match.group(1)) if year_match else None,
            "genres": genres,
            "avg_rating": self._avg_rating.get(int(movie_id)),
        }

    def movies(self, limit: int = 50, search: str | None = None) -> list[dict]:
        self.ensure_data()
        df = self._rec.movies
        if search:
            df = df[df["title"].str.contains(search, case=False, na=False)]
        return [self.movie_info(int(mid)) for mid in df["movie_id"].head(limit)]

    def recommend(self, user_id: int, algorithm: str, n: int) -> list[dict]:
        if algorithm not in ALGORITHMS:
            raise KeyError(algorithm)
        if self._status[algorithm] != Status.READY:
            raise RuntimeError(f"Model '{algorithm}' is not trained yet.")

        self.ensure_data()
        if algorithm == "SVD":
            raw = self._rec.get_svd_recommendations(user_id, n)
        elif algorithm == "NMF":
            raw = self._rec.get_nmf_recommendations(user_id, n)
        else:
            raw = self._rec.get_neural_recommendations(user_id, n)

        if raw is None:
            raise ValueError(f"User {user_id} not found.")

        results = []
        for r in raw:
            info = self.movie_info(int(r["movie_id"]))
            info["predicted_rating"] = round(float(r["predicted_rating"]), 2)
            results.append(info)
        return results


# Process-wide singleton.
registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return registry
