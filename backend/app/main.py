import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.config import Settings
from app.core.resources import open_resources

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            configuration = settings if settings is not None else Settings()
        except ValidationError:
            raise RuntimeError(
                "Invalid application configuration; check environment settings"
            ) from None
        async with open_resources(configuration) as resources:
            application.state.resources = resources
            yield

    application = FastAPI(
        title="Shopper Analytics",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Invalid request"})

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.error("Unhandled request failure", extra={"error_type": type(error).__name__})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        available = await request.app.state.resources.ready()
        return JSONResponse(
            status_code=200 if available else 503,
            content={"status": "ready" if available else "unavailable"},
        )

    return application


app = create_app()
