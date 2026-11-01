from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Standalone FUMOCA Frame Intelligence API",
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(jobs_router, prefix=settings.api_prefix)


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }
