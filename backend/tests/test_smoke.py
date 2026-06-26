"""Lightweight backend smoke tests.

These avoid the dataset download (no warm-up, no data-loading endpoints) so they
run fast in CI without network access.
"""

import os

os.environ.setdefault("AUTO_TRAIN_ON_STARTUP", "false")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

# Plain TestClient (no context manager) does not trigger lifespan/warm-up.
client = TestClient(main.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/health").status_code == 200


def test_algorithms_listed_without_training():
    data = client.get("/api/algorithms").json()
    keys = {a["key"] for a in data}
    assert {"SVD", "NMF", "Neural"} <= keys
    # Nothing trained yet -> no metrics.
    assert all(a["metrics"] is None for a in data)


def test_recommend_before_training_returns_409():
    resp = client.get("/api/recommend?user_id=1&algorithm=SVD&n=3")
    assert resp.status_code == 409
