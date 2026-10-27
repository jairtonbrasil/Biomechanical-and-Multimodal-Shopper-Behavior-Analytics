from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import Shelf
from app.repositories.assets import CameraRepository
from app.schemas.runtime import RuntimeConfiguration
from app.services.configuration import configuration_snapshot

router = APIRouter(prefix="/api/v1/cameras", tags=["configuration"])


@router.get("/{identifier}/runtime", response_model=RuntimeConfiguration)
async def get_configuration(
    identifier: UUID,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db, scope="function")],
) -> Response | RuntimeConfiguration:
    camera = await CameraRepository(session).get(identifier)
    shelves = list(
        await session.scalars(select(Shelf).where(Shelf.camera_id == identifier).order_by(Shelf.id))
    )
    resources = request.app.state.resources
    configuration = await configuration_snapshot(camera, shelves, resources.references)
    synchronizer = resources.configuration
    if synchronizer is not None:
        configuration = await synchronizer.status(configuration)
    etag = f'"{configuration.configuration_revision}:{configuration.proxy_state}"'
    if request.headers.get("if-none-match") == etag:
        return Response(
            status_code=304, headers={"ETag": etag, "Cache-Control": "private, no-store"}
        )
    return Response(
        content=configuration.model_dump_json(),
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "private, no-store"},
    )
