"""Filesystem-based API discovery.

Scans ``app/api/**/router.py``. Each such file must expose a variable named
``router`` that is an ``APIRouter``. The path below ``app/api`` becomes the URL
prefix, e.g.::

    app/api/recommend/router.py   -> /api/recommend
    app/api/train/router.py       -> /api/train

New route groups can be added without editing ``main.py``.
"""

import importlib
from pathlib import Path

from fastapi import APIRouter, FastAPI

API_DIR = Path(__file__).parent / "app" / "api"


def register_routes(app: FastAPI) -> list[str]:
    registered: list[str] = []
    for router_file in sorted(API_DIR.rglob("router.py")):
        rel_parts = router_file.relative_to(API_DIR).parent.parts
        module_name = ".".join(("app", "api", *rel_parts, "router"))
        module = importlib.import_module(module_name)
        router = getattr(module, "router", None)
        if not isinstance(router, APIRouter):
            continue
        prefix = "/api" + "".join(f"/{part}" for part in rel_parts)
        app.include_router(router, prefix=prefix)
        registered.append(prefix)
    return registered
