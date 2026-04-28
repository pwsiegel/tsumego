"""FastAPI application: assembles all domain routers."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from .health.routes import router as health_router
from .health.routes import start_warming
from .pdf.routes import router as pdf_router
from .study.routes import router as study_router
from .tsumego.routes import router as tsumego_router
from .val.routes import router as val_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_warming()
    yield


app = FastAPI(title="Go Problem Workbook API", version=__version__, lifespan=lifespan)

app.include_router(health_router)
app.include_router(pdf_router)
app.include_router(study_router)
app.include_router(tsumego_router)
app.include_router(val_router)


class SPAStaticFiles(StaticFiles):
    """StaticFiles with a single-page-app fallback: a 404 for a
    non-asset path (no file extension in the last segment) is served the
    bundled `index.html` so client-side routes like `/teacher` resolve
    on hard reload. Asset-shaped paths (`foo.js`, `bar.png`, …) still
    surface a real 404 so missing-asset bugs aren't masked as HTML."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404:
                raise
            last = path.rsplit("/", 1)[-1]
            if "." in last:
                raise
            return await super().get_response("index.html", scope)


# Serve the built frontend at `/`. In production the Dockerfile copies
# web/dist into $GOAPP_FRONTEND_DIR (default /app/frontend); locally it's
# unset and we skip the mount so `vite dev` on :5173 stays authoritative.
_frontend_dir = os.environ.get("GOAPP_FRONTEND_DIR")
if _frontend_dir and Path(_frontend_dir).is_dir():
    app.mount("/", SPAStaticFiles(directory=_frontend_dir, html=True), name="frontend")
