# Advanced Movie Recommendation System

A full-stack movie recommender built around three machine-learning algorithms
(SVD, NMF, Neural Collaborative Filtering) trained on the real **MovieLens 100K**
dataset. The recommendation engine, the API, and the UI are cleanly separated.

## 🏗️ Architecture

The project is split into three layers so the **core ML logic stays independent**
of any web framework:

```text
┌─────────────────────────┐     /api/* (same-origin)     ┌──────────────────────────┐
│        frontend         │  ───────────────────────►    │         backend          │
│  React (CRA) + nginx    │   ◄───────────────────────   │   FastAPI (REST API)     │
│  calls /api/* via proxy │        JSON responses        │  app/api/* + registry    │
└─────────────────────────┘                              └────────────┬─────────────┘
                                                                       │ imports
                                                                       ▼
                                                          ┌──────────────────────────┐
                                                          │           core           │
                                                          │  recommender/ (SVD, NMF, │
                                                          │  Neural CF) — no web deps │
                                                          └──────────────────────────┘
```

```text
movie-recommender/
├── core/        # the ML engine — installable Python package (the "main logic")
├── backend/     # FastAPI service that wraps the engine and exposes REST endpoints
├── frontend/    # React (Create React App) UI that calls the backend
└── docker-compose.yml
```

- **`core/`** — `AdvancedMovieRecommender` and the dataset loader. Pure Python; no
  FastAPI, no React. Runnable on its own via `python -m recommender`.
  See [core/README.md](core/README.md).
- **`backend/`** — FastAPI app. Trains and caches models in memory, exposes
  `/api/*`. Stateless (no database). See [backend/README.md](backend/README.md).
- **`frontend/`** — the React UI. It no longer holds any fake data; every movie,
  metric, and recommendation comes from the backend.

## 🚀 Quick Start (Docker — recommended)

```bash
docker compose up --build
```

- Frontend: <http://localhost:8080>
- Backend API + Swagger docs: <http://localhost:8000/docs>

On first start the backend downloads MovieLens 100K and trains the SVD + NMF
models in the background (a few seconds). The dataset is cached in a Docker
volume so subsequent starts are instant.

## 🛠️ Quick Start (local dev)

**Backend** — Python 3.10–3.12 recommended (3.13/3.14 also work for SVD + NMF;
the optional Neural CF extra needs a TensorFlow-supported version, currently ≤3.12):

```bash
pip install -e core            # ML engine (add "core[neural]" for TensorFlow)
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --reload    # http://localhost:8000
```

> Use `python -m uvicorn` rather than a bare `uvicorn`. The module form always
> uses your current interpreter and avoids a stale `uvicorn.exe` launcher left
> behind by a previous Python install (a common "cannot find the file specified"
> error after upgrading Python).

**Frontend** (Node 18+), in a second terminal:

```bash
cd frontend
npm install
npm start                      # http://localhost:3000
```

The frontend's `package.json` sets `"proxy": "http://localhost:8000"`, so the
browser calls `/api/*` on the same origin and CRA forwards them to the backend.

## 🎬 Using the app

1. Open the UI — <http://localhost:3000> (local dev) or <http://localhost:8080>
   (Docker). The three algorithm cards (SVD, NMF, Neural CF) show each model's
   status and, once trained, its RMSE/MAE.
2. The backend auto-trains **SVD** and **NMF** on startup, so they flip to
   **Ready** within a few seconds (first run also downloads the dataset). Click
   **Train All Models** to retrain, or to train Neural CF on demand — the status
   badges and progress panel update live while training runs.
3. Choose a **User ID** and how many **recommendations** you want from the
   dropdowns.
4. Click an algorithm card to switch methods. Recommendations refresh instantly,
   each showing the movie's title, year, genres, average rating, and the model's
   **predicted rating** for that user.
5. The **Model Performance Comparison** panel at the bottom contrasts RMSE/MAE
   across the trained models (lower is better).

> **Neural CF** shows as *Unavailable* when the backend has no TensorFlow (e.g.
> Python 3.13+). SVD and NMF work regardless. To enable Neural CF, run the
> backend on Python ≤3.12 with `pip install -e "core[neural]"`, or use the Docker
> backend image.

## 🔌 API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health`, `/api/ready` | Liveness/readiness probes (unauthenticated) |
| GET | `/api/algorithms` | Algorithms with status and current RMSE/MAE |
| GET | `/api/users` | Valid user IDs from the dataset |
| GET | `/api/movies?limit=&search=` | Movie catalog (title, year, genres, avg rating) |
| POST | `/api/train` | Train models — body `{"algorithm": "all"\|"SVD"\|"NMF"\|"Neural"}` |
| GET | `/api/train/status` | Per-model training status + metrics |
| GET | `/api/evaluate?algorithm=` | RMSE/MAE on the held-out test set |
| GET | `/api/recommend?user_id=&algorithm=&n=` | Recommendations for a user |

Interactive docs are served at `/docs` (Swagger) and `/redoc`.

## ⚙️ Configuration

| Variable | Default | Applies to | Purpose |
| --- | --- | --- | --- |
| `MOVIELENS_DATA_DIR` | `.` | core/backend | Where `ml-100k/` is downloaded/read |
| `AUTO_TRAIN_ON_STARTUP` | `true` | backend | Train SVD+NMF in the background at startup |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | backend | Comma-separated CORS origins |
| `SVD_COMPONENTS` / `NMF_COMPONENTS` | `50` | backend | Matrix-factorization dimensionality |
| `NEURAL_EMBEDDING_SIZE` / `NEURAL_EPOCHS` | `50` / `20` | backend | Neural CF hyperparameters |
| `REACT_APP_API_URL` | `""` (same origin) | frontend | Override backend base URL at build time |

## 📊 Algorithms

- **SVD Matrix Factorization** — `TruncatedSVD` on the mean-centered user-item
  matrix. Good general-purpose collaborative filtering.
- **NMF** — non-negative factorization; more interpretable factors.
- **Neural Collaborative Filtering** — user/item embeddings + dense layers
  (TensorFlow). Optional and **lazy**: it is only trained on request, and only if
  TensorFlow is installed (`pip install -e "core[neural]"`). The UI marks it
  *Unavailable* otherwise.

## 🧪 Tests

```bash
cd backend && pytest tests/      # backend smoke tests (no dataset download)
cd frontend && CI=true npm test  # frontend component test
```

## 📝 Notes

- The engine computes user means over a zero-filled user-item matrix (inherited
  from the original implementation), so reported RMSE is higher than the headline
  numbers a fully tuned recommender would reach. Tuning model quality is tracked
  as a separate improvement and intentionally left out of this restructure.

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

*Built with React, FastAPI, scikit-learn, TensorFlow, and ❤️ for machine learning*
