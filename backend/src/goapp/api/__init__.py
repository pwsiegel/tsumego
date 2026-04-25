"""FastAPI application: assembles all domain routers."""

import logging

from fastapi import FastAPI

from .. import __version__
from .health.routes import router as health_router
from .pdf.routes import router as pdf_router
from .tsumego.routes import router as tsumego_router
from .val.routes import router as val_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Go Problem Workbook API", version=__version__)

app.include_router(health_router)
app.include_router(pdf_router)
app.include_router(tsumego_router)
app.include_router(val_router)
