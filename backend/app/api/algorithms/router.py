"""GET /api/algorithms — list available algorithms and their current metrics."""

from fastapi import APIRouter, Depends

from app.schemas import Algorithm
from app.services.registry import ModelRegistry, get_registry

router = APIRouter(tags=["algorithms"])


@router.get("", response_model=list[Algorithm])
def list_algorithms(registry: ModelRegistry = Depends(get_registry)):
    return registry.algorithms()
