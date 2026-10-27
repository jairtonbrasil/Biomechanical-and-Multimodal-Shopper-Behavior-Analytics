import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from redis.exceptions import RedisError
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.config import Settings
from app.core.references import ReferenceUnavailableError
from app.core.resources import open_resources
from app.middleware.limits import OperationalLimits
from app.repositories.errors import EntityNotFoundError, InvalidReferenceError, VersionConflictError
from app.routers.assets import router as asset_router
from app.routers.playback import router as playback_router
from app.routers.runtime import router as runtime_router

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
            application.state.settings = configuration
            application.state.resources = resources
            yield

    application = FastAPI(
        title="Shopper Analytics",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if settings is not None:
        application.state.settings = settings
    application.add_middleware(OperationalLimits)
    application.include_router(asset_router)
    application.include_router(runtime_router)
    application.include_router(playback_router)
    application.mount(
        "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
    )

    async def domain_error(request: Request, error: Exception) -> JSONResponse:
        if isinstance(error, EntityNotFoundError):
            code, detail = 404, "Resource does not exist"
        elif isinstance(error, (InvalidReferenceError, VersionConflictError, IntegrityError)):
            code, detail = 409, "Configuration conflict; reload the current state"
        elif isinstance(error, (ReferenceUnavailableError, OperationalError, RedisError)):
            code, detail = 503, "Requested service or reference is unavailable"
        else:
            code, detail = 422, "Invalid configuration"
        return JSONResponse(status_code=code, content={"detail": detail})

    for error_type in (
        EntityNotFoundError,
        InvalidReferenceError,
        VersionConflictError,
        IntegrityError,
        ReferenceUnavailableError,
        OperationalError,
        RedisError,
        ValueError,
    ):
        application.add_exception_handler(error_type, domain_error)

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
