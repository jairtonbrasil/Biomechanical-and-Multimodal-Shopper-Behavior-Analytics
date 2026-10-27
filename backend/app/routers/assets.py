from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.stream_policy import StreamPolicy
from app.schemas.assets import (
    CameraCreate,
    CameraResponse,
    CameraUpdate,
    ShelfCreate,
    ShelfResponse,
    ShelfUpdate,
)
from app.schemas.references import ReferenceFrame
from app.services.assets import CameraService, ShelfService

router = APIRouter(prefix="/api/v1", tags=["configuration"])
Session = Annotated[AsyncSession, Depends(get_db, scope="function")]
Offset = Annotated[int, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=100)]


def camera_service(request: Request, session: AsyncSession) -> CameraService:
    return CameraService(
        session,
        StreamPolicy(request.app.state.settings),
        request.app.state.resources.references,
    )


def reference_response(reference: ReferenceFrame) -> Response:
    return Response(
        content=reference.content,
        media_type="image/jpeg",
        headers={
            "ETag": f'"{reference.revision}"',
            "Cache-Control": "private, no-store",
            "X-Reference-Revision": str(reference.revision),
            "X-Source-Revision": str(reference.source_revision),
            "X-Frame-Width": str(reference.width),
            "X-Frame-Height": str(reference.height),
            "X-Captured-At": reference.captured_at.isoformat(),
        },
    )


@router.post("/cameras", response_model=CameraResponse, status_code=201)
async def create_camera(data: CameraCreate, request: Request, session: Session) -> CameraResponse:
    camera = await camera_service(request, session).create(data)
    return CameraResponse.model_validate(camera)


@router.get("/cameras", response_model=list[CameraResponse])
async def list_cameras(
    request: Request,
    session: Session,
    offset: Offset = 0,
    limit: Limit = 100,
    is_active: bool | None = None,
) -> list[CameraResponse]:
    cameras = await camera_service(request, session).list(
        offset=offset, limit=limit, is_active=is_active
    )
    return [CameraResponse.model_validate(camera) for camera in cameras]


@router.get("/cameras/{identifier}", response_model=CameraResponse)
async def get_camera(identifier: UUID, request: Request, session: Session) -> CameraResponse:
    camera = await camera_service(request, session).get(identifier)
    return CameraResponse.model_validate(camera)


@router.put("/cameras/{identifier}", response_model=CameraResponse)
async def update_camera(
    identifier: UUID,
    data: CameraUpdate,
    request: Request,
    session: Session,
) -> CameraResponse:
    camera = await camera_service(request, session).update(identifier, data)
    return CameraResponse.model_validate(camera)


@router.get("/cameras/{identifier}/frame-zero")
async def get_reference(identifier: UUID, request: Request, session: Session) -> Response:
    reference = await camera_service(request, session).reference(identifier)
    return reference_response(reference)


@router.post("/cameras/{identifier}/frame-zero")
async def refresh_reference(identifier: UUID, request: Request, session: Session) -> Response:
    reference = await camera_service(request, session).reference(identifier, refresh=True)
    return reference_response(reference)


@router.post("/shelves", response_model=ShelfResponse, status_code=201)
async def create_shelf(data: ShelfCreate, request: Request, session: Session) -> ShelfResponse:
    shelf = await ShelfService(session, request.app.state.resources.references).create(data)
    return ShelfResponse.model_validate(shelf)


@router.get("/shelves", response_model=list[ShelfResponse])
async def list_shelves(
    request: Request,
    session: Session,
    offset: Offset = 0,
    limit: Limit = 100,
    camera_id: UUID | None = None,
    is_active: bool | None = None,
) -> list[ShelfResponse]:
    shelves = await ShelfService(session, request.app.state.resources.references).list(
        offset=offset, limit=limit, camera_id=camera_id, is_active=is_active
    )
    return [ShelfResponse.model_validate(shelf) for shelf in shelves]


@router.get("/shelves/{identifier}", response_model=ShelfResponse)
async def get_shelf(identifier: UUID, request: Request, session: Session) -> ShelfResponse:
    shelf = await ShelfService(session, request.app.state.resources.references).get(identifier)
    return ShelfResponse.model_validate(shelf)


@router.put("/shelves/{identifier}", response_model=ShelfResponse)
async def update_shelf(
    identifier: UUID,
    data: ShelfUpdate,
    request: Request,
    session: Session,
) -> ShelfResponse:
    shelf = await ShelfService(session, request.app.state.resources.references).update(
        identifier, data
    )
    return ShelfResponse.model_validate(shelf)
