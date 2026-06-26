"""GET /api/users — valid user IDs from the dataset."""

from fastapi import APIRouter, Depends

from app.services.registry import ModelRegistry, get_registry

router = APIRouter(tags=["users"])


@router.get("")
def list_users(registry: ModelRegistry = Depends(get_registry)):
    users = registry.users()
    return {"count": len(users), "users": users}
