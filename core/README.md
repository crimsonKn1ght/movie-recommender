# Movie Recommender — Core Engine

The **main recommendation logic**, kept independent of any web framework. This
package implements three algorithms over the MovieLens 100K dataset:

- **SVD** matrix factorization (`TruncatedSVD`)
- **NMF** non-negative matrix factorization
- **Neural Collaborative Filtering** (optional, requires TensorFlow)

The FastAPI backend in [`../backend`](../backend) imports
`AdvancedMovieRecommender` from this package and exposes its results over HTTP.
Nothing here depends on FastAPI.

## Layout

```text
core/
├── recommender/
│   ├── __init__.py      # public exports
│   ├── data.py          # MovieLens download / load
│   ├── engine.py        # AdvancedMovieRecommender (the algorithms)
│   └── __main__.py      # CLI demo
└── pyproject.toml
```

## Install

```bash
pip install -e core            # SVD + NMF
pip install -e "core[neural]"  # also installs TensorFlow for Neural CF
```

## Run the CLI demo

```bash
python -m recommender
```

This downloads the dataset (first run only), trains the models, and prints
sample recommendations and evaluation metrics.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `MOVIELENS_DATA_DIR` | `.` (CWD) | Directory where `ml-100k/` is downloaded/read. Point this at a mounted volume or a path baked into the container image. |
