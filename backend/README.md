# Movie Recommender — Backend

FastAPI service that wraps the [`core`](../core) recommendation engine and
exposes it over REST. It is **stateless**: models are trained and cached in
memory for the life of the process; there is no database and no auth.

## Layout

```text
backend/
├── main.py              # app factory: lifespan, health probes, CORS, router loading
├── router_loader.py     # filesystem discovery of app/api/**/router.py
├── app/
│   ├── api/             # one folder per route group -> /api/<folder>
│   │   ├── algorithms/  ├── users/   ├── movies/
│   │   ├── train/       ├── evaluate/└── recommend/
│   ├── services/registry.py   # in-memory ModelRegistry (trains, caches, evaluates)
│   ├── schemas/         # pydantic request/response models
│   └── utils/settings.py
├── tests/               # smoke tests (no dataset download)
└── requirements.txt
```

## How it works

- **Dynamic routing** — `router_loader.register_routes()` scans
  `app/api/**/router.py`; the path under `app/api` becomes the URL prefix
  (`app/api/recommend/router.py` → `/api/recommend`). Add a new route group by
  dropping in a folder with a `router.py` that exposes an `APIRouter` named
  `router`; no edit to `main.py` required.
- **Model registry** — `app/services/registry.py` owns a single
  `AdvancedMovieRecommender`. SVD/NMF are trained eagerly in a background thread
  at startup; Neural CF is trained lazily on `POST /api/train`. Training is
  guarded by a lock and progress is exposed via `GET /api/train/status` so the UI
  can poll real state.

## Run

```bash
pip install -e ../core
pip install -r requirements.txt
uvicorn main:app --reload
```

See the [root README](../README.md) for the full endpoint and configuration
reference.

## Test

```bash
pip install -r requirements-dev.txt   # pytest + httpx (for TestClient)
AUTO_TRAIN_ON_STARTUP=false pytest tests/
```
