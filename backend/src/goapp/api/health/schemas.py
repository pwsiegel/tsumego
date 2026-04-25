from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str  # "warming" | "ready" | "degraded"
    version: str
    models_ready: bool
    error: str | None = None
