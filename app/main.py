import secrets

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.utils import get_openapi

from app.api.routes.analysis import router as analysis_router
from app.api.routes.health import router as health_router
from app.api.routes.inspection import router as inspection_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.matching import router as matching_router
from app.core.config import settings
from app.web.studio import STUDIO_HTML

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="Standalone FUMOCA Frame Intelligence API",
)

@app.middleware("http")
async def protect_non_development_api(request: Request, call_next):
    environment = settings.environment.lower()
    if environment != "development":
        if not settings.api_key:
            return JSONResponse(
                status_code=503,
                content={"detail": "API is disabled: configure IMAGE_CREATE_API_KEY before non-development use."},
            )
        if request.url.path.startswith(settings.api_prefix):
            supplied = request.headers.get("X-API-Key") or request.cookies.get("image_create_session", "")
            if not secrets.compare_digest(supplied, settings.api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "A valid X-API-Key header is required."},
                    headers={"WWW-Authenticate": "ApiKey"},
                )
    return await call_next(request)


@app.post(f"{settings.api_prefix}/session", tags=["browser session"])
def create_browser_session(request: Request, response: Response) -> dict[str, str]:
    """Exchange the shared API key for a same-site browser-session cookie."""
    if settings.environment.lower() == "development":
        return {"status": "development_no_session_required"}
    if not settings.api_key:
        raise HTTPException(
            status_code=503,
            detail="Browser session is disabled until IMAGE_CREATE_API_KEY is configured.",
        )
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied, settings.api_key):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
    response.set_cookie(
        key="image_create_session",
        value=settings.api_key,
        httponly=True,
        secure=True,
        samesite="strict",
        path=settings.api_prefix,
    )
    return {"status": "authenticated", "scope": "browser_session"}


app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(jobs_router, prefix=settings.api_prefix)
app.include_router(analysis_router, prefix=settings.api_prefix)
app.include_router(matching_router, prefix=settings.api_prefix)
app.include_router(inspection_router, prefix=settings.api_prefix)


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


@app.get("/", tags=["root"], response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Serve the browser-based capture upload and inspection entry point."""
    return HTMLResponse(STUDIO_HTML)
