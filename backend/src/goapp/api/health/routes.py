"""Health check endpoint with model-warming status.

The container starts immediately and accepts HTTP traffic, but the YOLO
models take ~5-10s to load. A background thread kicks off the load at
startup; the frontend polls this endpoint and waits for status='ready'
before rendering the app.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter

from ... import __version__
from .schemas import HealthResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_state: dict[str, object] = {"ready": False, "error": None}
_lock = threading.Lock()


def _warm_models() -> None:
    try:
        from ...ml.board_detect.detect import warm as warm_board
        from ...ml.stone_detect.detect import warm as warm_stones
        warm_board()
        warm_stones()
        with _lock:
            _state["ready"] = True
        log.info("models warmed: board + stone detectors loaded")
    except Exception as e:
        log.exception("model warming failed")
        with _lock:
            _state["error"] = str(e)


def start_warming() -> None:
    """Kick off model loading in a background thread (idempotent)."""
    with _lock:
        if _state.get("started"):
            return
        _state["started"] = True
    threading.Thread(target=_warm_models, name="model-warmer", daemon=True).start()


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    with _lock:
        ready = bool(_state.get("ready"))
        err = _state.get("error")
    if err:
        status = "degraded"
    elif ready:
        status = "ready"
    else:
        status = "warming"
    return HealthResponse(
        status=status,
        version=__version__,
        models_ready=ready,
        error=err if isinstance(err, str) else None,
    )
