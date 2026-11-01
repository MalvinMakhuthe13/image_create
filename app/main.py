from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.matching import router as matching_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="Standalone FUMOCA Frame Intelligence API",
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(jobs_router, prefix=settings.api_prefix)
app.include_router(analysis_router, prefix=settings.api_prefix)
app.include_router(matching_router, prefix=settings.api_prefix)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Make uploaded files recognizable to Swagger UI.
    upload_schema = (
        schema.get("components", {})
        .get("schemas", {})
        .get("Body_analyse_images_api_v1_analysis_images_post", {})
    )

    files_schema = upload_schema.get("properties", {}).get("files", {})
    items_schema = files_schema.get("items", {})

    if items_schema.get("type") == "string":
        items_schema["format"] = "binary"
        items_schema.pop("contentMediaType", None)

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }
