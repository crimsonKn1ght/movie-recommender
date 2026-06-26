"""GET /api/evaluate — RMSE/MAE for trained models on the held-out test set."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.registry import ALGORITHMS, ModelRegistry, get_registry

router = APIRouter(tags=["evaluate"])


@router.get("")
def evaluate(
    algorithm: str | None = Query(None),
    registry: ModelRegistry = Depends(get_registry),
):
    status = registry.status()["models"]

    if algorithm is not None:
        if algorithm not in ALGORITHMS:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")
        metrics = status[algorithm]["metrics"]
        if metrics is None:
            raise HTTPException(
                status_code=409, detail=f"Model '{algorithm}' has no metrics yet."
            )
        return {"algorithm": algorithm, "metrics": metrics}

    return {key: status[key]["metrics"] for key in ALGORITHMS}
