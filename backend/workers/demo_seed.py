"""Register an aligned local demonstration without running the inference pipeline."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from sqlalchemy import select

from app.core.config import Settings
from app.core.database import Database
from app.core.references import ReferenceStore
from app.core.stream_policy import StreamPolicy
from app.domain.configuration import source_revision
from app.models import Camera
from app.repositories.assets import CameraRepository, ShelfRepository
from app.schemas.assets import CameraCreate, GeometryUpdate, ShelfCreate
from app.schemas.geometry import Polygon, ReferencedPolygon


def shelf_specifications(path: Path) -> list[dict]:
    specifications = json.loads(path.read_text())
    if not specifications or len(specifications) > 32:
        raise ValueError("A demo needs between one and 32 shelf definitions")
    names = set()
    for item in specifications:
        if item["name"] in names:
            raise ValueError("Demo shelf names must be unique")
        names.add(item["name"])
        Polygon.model_validate(item["product_roi"])
        Polygon.model_validate(item["presence_roi"])
        ShelfCreate.model_validate(
            {
                **item,
                "camera_id": UUID(int=1),
                "product_roi": {**item["product_roi"], "reference_revision": UUID(int=1)},
            }
        )
    return specifications


async def seed(
    settings: Settings, scenario: dict, specifications: list[dict], image: bytes
) -> dict:
    password = quote(settings.mediamtx_read_password.get_secret_value(), safe="")
    stream = f"rtsp://media-reader:{password}@127.0.0.1:8554/{scenario['stream_path']}"
    StreamPolicy(settings).validate(stream)
    store, database = ReferenceStore(settings), Database(settings)
    try:
        async with database.transaction() as session:
            cameras, shelves = CameraRepository(session), ShelfRepository(session)
            matches = list(
                await session.scalars(
                    select(Camera).where(Camera.name == scenario["camera_name"]).with_for_update()
                )
            )
            if len(matches) > 1:
                raise ValueError("Duplicate scenario cameras require manual review")
            camera = (
                matches[0]
                if matches
                else await cameras.create(
                    CameraCreate(
                        name=scenario["camera_name"],
                        stream_url=stream,
                        is_active=True,
                    )
                )
            )
            if camera.stream_url != stream:
                raise ValueError("An existing scenario camera has a different source")
            camera.is_active = True
            current = await asyncio.to_thread(store.read, camera.id, source_revision(camera))
            existing = await cameras.shelves(camera.id, lock=True)
            if current is None:
                if any(item.is_active for item in existing):
                    raise ValueError("Do not replace a missing reference under active geometry")
                current = await asyncio.to_thread(
                    store.publish, camera.id, source_revision(camera), image
                )
            for specification in specifications:
                product = ReferencedPolygon(
                    **specification["product_roi"], reference_revision=current.revision
                )
                presence = Polygon.model_validate(specification["presence_roi"])
                matching = [item for item in existing if item.name == specification["name"]]
                if len(matching) > 1:
                    raise ValueError("Duplicate scenario shelves require manual review")
                if not matching:
                    await shelves.create(
                        ShelfCreate(
                            **{
                                **specification,
                                "camera_id": camera.id,
                                "product_roi": product,
                                "presence_roi": presence,
                            }
                        )
                    )
                else:
                    item = matching[0]
                    accepted = {key: item.product_roi[key] for key in type(product).model_fields}
                    if accepted != product.model_dump(
                        mode="json"
                    ) or item.presence_roi != presence.model_dump(mode="json"):
                        await shelves.update_geometry(
                            item.id,
                            GeometryUpdate(
                                expected_version=item.roi_version,
                                product_roi=product,
                                presence_roi=presence,
                                cutover_at=datetime.now(UTC),
                            ),
                        )
                    item.category, item.is_active = specification["category"], True
            # Retire only the previous reserved demo, preserving all records and evidence.
            legacy = list(
                await session.scalars(
                    select(Camera).where(Camera.name == "Stage 3 Retail Demo").with_for_update()
                )
            )
            for previous in legacy:
                if await cameras.has_active_interactions(previous.id):
                    raise ValueError("Close legacy demo interactions before switching scenarios")
                previous.is_active = False
                for shelf in await cameras.shelves(previous.id, lock=True):
                    shelf.is_active = False
            await session.flush()
            result = {
                "camera_id": str(camera.id),
                "camera_name": camera.name,
                "reference_revision": str(current.revision),
                "video_file": scenario["video_file"],
                "shelf_count": len(specifications),
                "shelves": [
                    {"name": item["name"], "category": item["category"]} for item in specifications
                ],
                "legacy_records_preserved": True,
            }
        return result
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--shelves", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        seed(
            Settings(),
            json.loads(args.scenario.read_text()),
            shelf_specifications(args.shelves),
            args.reference.read_bytes(),
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
