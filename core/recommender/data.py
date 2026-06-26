"""MovieLens 100K dataset access.

This module owns everything about *getting the data onto disk and into
DataFrames*. It is intentionally free of any modelling logic so the engine can
depend on it without pulling in scikit-learn or TensorFlow concerns.

The data directory is configurable through the ``MOVIELENS_DATA_DIR``
environment variable (default: current working directory). This lets the
backend point downloads at a mounted volume or a path baked into a container
image instead of scattering ``ml-100k/`` into the process CWD.
"""

import os
import zipfile

import pandas as pd
import requests

MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Column layouts for the raw MovieLens 100K files.
RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLUMNS = [
    "movie_id", "title", "release_date", "video_release_date",
    "imdb_url", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def get_data_dir() -> str:
    """Return the directory that should contain the ``ml-100k`` dataset."""
    return os.environ.get("MOVIELENS_DATA_DIR", ".")


def download_movielens_data(data_dir: str | None = None) -> str:
    """Download and extract the MovieLens 100K dataset.

    Returns the path to the extracted ``ml-100k`` directory.
    """
    data_dir = data_dir or get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")

    print("Downloading MovieLens 100K dataset...")
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset downloaded and extracted!")
    return os.path.join(data_dir, "ml-100k")


def load_ratings_and_movies(data_dir: str | None = None):
    """Load the ratings and movies DataFrames, downloading first if needed."""
    data_dir = data_dir or get_data_dir()
    dataset_path = os.path.join(data_dir, "ml-100k")

    if not os.path.exists(dataset_path):
        download_movielens_data(data_dir)

    ratings = pd.read_csv(
        os.path.join(dataset_path, "u.data"),
        sep="\t",
        names=RATINGS_COLUMNS,
    )

    movies = pd.read_csv(
        os.path.join(dataset_path, "u.item"),
        sep="|",
        encoding="latin-1",
        names=MOVIES_COLUMNS,
    )

    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
    return ratings, movies
