from datetime import timedelta
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.core.database import Database
from app.models import Base, ShelfMetric, ShelfSignalMetric
from app.repositories.assets import ShelfRepository
from app.repositories.events import EventRepository
from app.repositories.observations import ObservationRepository
from app.schemas.assets import GeometryUpdate
from app.schemas.common import LinkType, ModelKind
from app.schemas.events import ActionCreate, AttentionCreate, ReactionCreate
from app.schemas.geometry import StoredProductROI
from tests.conftest import migration_config
from tests.factories import POLYGON, observation
from tests.test_contracts import action_data
from tests.test_persistence import assets, interaction, versions

pytestmark = pytest.mark.integration


async def snapshot_rows(connection) -> dict[str, list[dict]]:
    result = {}
    for table in sorted(Base.metadata.tables):
        rows = await connection.scalars(text(f'SELECT to_jsonb(t) FROM "{table}" t'))
        result[table] = sorted(rows, key=str)
    return result


async def test_english_migration_preserves_records_snapshots_and_scoped_ids(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        identifiers = await versions(session)
        old = ShelfRepository.snapshot(shelf)
        parent = await interaction(session, shelf, old.valid_from, active=False)
        row = await ObservationRepository(session).add(
            observation(old, identifiers, old.valid_from)
        )
        await ObservationRepository(session).link(row.id, parent.id, LinkType.PRESENCE)
        events = EventRepository(session)
        await events.add_attention(
            AttentionCreate(
                id=uuid4(),
                interaction_id=parent.id,
                model_id=identifiers[ModelKind.ATTENTION],
                started_at=old.valid_from,
                ended_at=old.valid_from + timedelta(seconds=1),
                method="head-cone",
                confidence=0.8,
            )
        )
        await events.add_action(
            ActionCreate.model_validate(
                {
                    **action_data(),
                    "interaction_id": parent.id,
                    "model_id": identifiers[ModelKind.ACTION],
                    "occurred_at": old.valid_from,
                }
            )
        )
        await events.add_reaction(
            ReactionCreate.model_validate(
                {
                    "id": uuid4(),
                    "interaction_id": parent.id,
                    "model_id": identifiers[ModelKind.BODY_REACTION],
                    "signal_type": "HESITATION",
                    "source": "BODY_BIOMECHANICS",
                    "occurred_at": old.valid_from,
                    "confidence": 0.7,
                    "attributes": {
                        "facial_analysis": {
                            "face_visible": False,
                            "detected_expression": None,
                            "visibility_quality": None,
                            "occlusion_estimated": None,
                        },
                        "behavioral_context": {
                            "person_standing": True,
                            "attention_detected": None,
                            "handling_product": None,
                        },
                        "temporal_evidence": {
                            "start_frame": 0,
                            "end_frame": 14,
                            "duration_ms": 1400,
                        },
                        "association": {"single_shelf_context": True, "context_quality": 0.8},
                        "inferred_emotion": None,
                    },
                }
            )
        )
        await ShelfRepository(session).update_geometry(
            shelf.id,
            GeometryUpdate(
                expected_version=1,
                product_roi={**POLYGON, "reference_revision": old.reference_revision},
                presence_roi=POLYGON,
                cutover_at=old.valid_from + timedelta(seconds=2),
            ),
        )
        session.add(
            ShelfMetric(
                shelf_id=shelf.id,
                granularity="hour",
                window_start=old.valid_from,
                window_end=old.valid_from + timedelta(hours=1),
                interactions_started=1,
            )
        )
        session.add(
            ShelfSignalMetric(
                shelf_id=shelf.id,
                granularity="hour",
                window_start=old.valid_from,
                signal_type="HESITATION",
                signal_count=1,
            )
        )
        shelf_id, observation_id = shelf.id, row.id

    async with domain_database.engine.begin() as connection:
        before = await snapshot_rows(connection)
        assert all(before.values()), "Every domain table must contain migration evidence"
        await connection.run_sync(
            lambda sync: command.downgrade(migration_config(sync), "0001_data_layer")
        )
        legacy = await connection.scalar(
            text("SELECT atributos FROM observacao_frame WHERE id = :id"), {"id": observation_id}
        )
        assert legacy["provenance"]["schema_version"] == "1.0"
        assert legacy["geometry"][0]["gondola_id"] == str(shelf_id)
        assert "gondola_confidence" in legacy["attention"]
        history = await connection.scalar(
            text("SELECT roi_produtos->'history' FROM gondola WHERE id = :id"), {"id": shelf_id}
        )
        assert history[0]["gondola_id"] == str(shelf_id)
        await connection.run_sync(lambda sync: command.upgrade(migration_config(sync), "head"))
        assert await snapshot_rows(connection) == before
        await connection.run_sync(lambda sync: command.check(migration_config(sync)))

    async with domain_database.transaction() as session:
        restored = await ShelfRepository(session).get(shelf_id)
        assert StoredProductROI.model_validate(restored.product_roi).history[0].shelf_id == shelf_id
        saved = await session.scalar(select(Base.metadata.tables["frame_observation"].c.attributes))
        assert saved["provenance"]["schema_version"] == "2.0"


async def test_key_collision_rolls_back_the_entire_naming_migration(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifiers = await versions(session)
        row = await ObservationRepository(session).add(
            observation(ShelfRepository.snapshot(shelf), identifiers, shelf.created_at)
        )
        identifier = row.id

    async with domain_database.engine.begin() as connection:
        await connection.run_sync(
            lambda sync: command.downgrade(migration_config(sync), "0001_data_layer")
        )
        await connection.execute(
            text("""
            UPDATE observacao_frame SET atributos = jsonb_set(atributos,
                '{geometry,0,shelf_id}', to_jsonb(CAST(:id AS text)))
        """),
            {"id": str(uuid4())},
        )
        with pytest.raises(IntegrityError, match="Conflicting contract keys"):
            async with connection.begin_nested():
                await connection.run_sync(
                    lambda sync: command.upgrade(migration_config(sync), "head")
                )
        assert await connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "0001_data_layer"
        )
        tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert "observacao_frame" in tables and "frame_observation" not in tables
        assert (
            await connection.scalar(
                text("SELECT id FROM observacao_frame WHERE id = :id"), {"id": identifier}
            )
            == identifier
        )
        await connection.execute(
            text("""
            UPDATE observacao_frame SET atributos = atributos #- '{geometry,0,shelf_id}'
        """)
        )
        await connection.run_sync(lambda sync: command.upgrade(migration_config(sync), "head"))
