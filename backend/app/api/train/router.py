"""POST /api/train and GET /api/train/status.

Training runs in a background daemon thread so the request returns immediately
and the frontend can poll ``/api/train/status`` to drive its progress UI with
real state. SVD/NMF are quick; Neural CF (if TensorFlow is installed) is slow,
which is exactly why it is kept off the request path.
"""

import threading

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import TrainRequest
from app.services.registry import ALGORITHMS, ModelRegistry, get_registry

router = APIRouter(tags=["train"])


def _train_async(registry: ModelRegistry, algorithms: list[str]) -> None:
    for algorithm in algorithms:
        try:
            registry.train(algorithm)
        except Exception:  # noqa: BLE001 - status/errors recorded on the registry
            pass


@router.post("", status_code=202)
def train(req: TrainRequest, registry: ModelRegistry = Depends(get_registry)):
    if req.algorithm == "all":
        algorithms = list(ALGORITHMS.keys())
    elif req.algorithm in ALGORITHMS:
        algorithms = [req.algorithm]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {req.algorithm}")

    threading.Thread(
        target=_train_async, args=(registry, algorithms), daemon=True
    ).start()
    return {"started": algorithms, "status": registry.status()}


@router.get("/status")
def train_status(registry: ModelRegistry = Depends(get_registry)):
    return registry.status()
