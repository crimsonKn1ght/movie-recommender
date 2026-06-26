"""GET /api/recommend — real recommendations from a trained model."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas import Recommendation
from app.services.registry import ModelRegistry, get_registry

router = APIRouter(tags=["recommend"])


@router.get("", response_model=list[Recommendation])
def recommend(
    user_id: int = Query(..., ge=1),
    algorithm: str = Query("SVD"),
    n: int = Query(5, ge=1, le=50),
    registry: ModelRegistry = Depends(get_registry),
):
    try:
        return registry.recommend(user_id=user_id, algorithm=algorithm, n=n)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
