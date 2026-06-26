"""GET /api/movies — the real MovieLens movie catalog."""

from fastapi import APIRouter, Depends, Query

from app.schemas import Movie
from app.services.registry import ModelRegistry, get_registry

router = APIRouter(tags=["movies"])


@router.get("", response_model=list[Movie])
def list_movies(
    limit: int = Query(50, ge=1, le=500),
    search: str | None = Query(None),
    registry: ModelRegistry = Depends(get_registry),
):
    return registry.movies(limit=limit, search=search)
