import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import delete, func, inspect, select, update
from sqlalchemy.exc import IntegrityError

from app.core.database import Database
from app.models import (
    Base,
    Camera,
    FrameObservation,
    ModelVersion,
    ObservationInteraction,
    ReactionSignal,
    Shelf,
    ShelfInteraction,
    ShelfMetric,
    ShelfSignalMetric,
)
from app.repositories.assets import CameraRepository, ShelfRepository
from app.repositories.errors import InvalidReferenceError, VersionConflictError
from app.repositories.events import EventRepository, InteractionRepository
from app.repositories.history import HistoryPurpose, HistoryRepository, VerifiedCoverage
from app.repositories.model_versions import ModelVersionRepository
from app.repositories.observations import ObservationRepository
from app.schemas.assets import (
    CameraCreate,
    CameraResponse,
    GeometryUpdate,
    ShelfCreate,
    ShelfResponse,
)
from app.schemas.common import LinkType, ModelKind, TimeWindow
from app.schemas.events import ActionCreate, AttentionCreate, InteractionCreate, ReactionCreate
from app.schemas.observation import ObservationCreate
from tests.conftest import migration_config
from tests.factories import POLYGON, model_version, observation
from tests.test_contracts import action_data

pytestmark = pytest.mark.integration


async def assets(session):
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="Test camera",
            stream_url="rtsp://test-user:test-password@camera.invalid/live",
            is_active=False,
        )
    )
    shelf = await ShelfRepository(session).create(
        ShelfCreate.model_validate(
            {
                "camera_id": camera.id,
                "name": "Test shelf",
                "category": "Test category",
                "product_roi": {**POLYGON, "reference_revision": uuid4()},
                "presence_roi": POLYGON,
            }
        )
    )
    return camera, shelf


async def versions(session):
    repository = ModelVersionRepository(session)
    identifiers = {}
    for kind in ModelKind:
        dependencies = {}
        if kind in {
            ModelKind.ATTENTION,
            ModelKind.ACTION,
            ModelKind.BODY_REACTION,
            ModelKind.FACE_BODY_REACTION,
        }:
            dependencies = {
                key: identifiers[key] for key in (ModelKind.POSE, ModelKind.TEMPORAL_RULES)
            }
        if kind == ModelKind.FACE_BODY_REACTION:
            dependencies[ModelKind.FACE] = identifiers[ModelKind.FACE]
        identifiers[kind] = (await repository.register(model_version(kind, dependencies))).id
    return identifiers


async def interaction(session, shelf, instant, *, active=True):
    return await InteractionRepository(session).create(
        InteractionCreate(
            id=uuid4(),
            shelf_id=shelf.id,
            started_at=instant,
            ended_at=None if active else instant + timedelta(seconds=1),
            last_presence_at=instant,
            presence_duration_ms=0,
            status="ACTIVE" if active else "CLOSED",
            end_reason=None if active else "LEFT_ROI",
            roi_version=shelf.roi_version,
        )
    )


async def test_migration_schema_matches_models_and_has_all_indexes(
    domain_database: Database,
) -> None:
    async with domain_database.engine.connect() as connection:

        def compare(sync):
            inspector = inspect(sync)
            tables = set(inspector.get_table_names()) - {"alembic_version"}
            assert tables == set(Base.metadata.tables)
            for name, model in Base.metadata.tables.items():
                assert set(model.columns.keys()) == {
                    item["name"] for item in inspector.get_columns(name)
                }
                assert set(model.primary_key.columns.keys()) == set(
                    inspector.get_pk_constraint(name)["constrained_columns"]
                )
                assert len(model.foreign_keys) == len(inspector.get_foreign_keys(name))
                for foreign_key in inspector.get_foreign_keys(name):
                    assert foreign_key["options"]["ondelete"] == "RESTRICT"
                actual_checks = {item["name"] for item in inspector.get_check_constraints(name)}
                assert {
                    item.name
                    for item in model.constraints
                    if item.__class__.__name__ == "CheckConstraint"
                } == actual_checks
            differences = compare_metadata(
                MigrationContext.configure(sync, opts={"compare_server_default": True}),
                Base.metadata,
            )
            assert not differences
            indexes = inspector.get_indexes("frame_observation")
            assert any(item["dialect_options"].get("postgresql_using") == "gin" for item in indexes)

        await connection.run_sync(compare)


async def test_migration_downgrade_and_upgrade_on_disposable_schema(
    domain_database: Database,
) -> None:
    async with domain_database.engine.begin() as connection:
        await connection.run_sync(lambda sync: command.downgrade(migration_config(sync), "base"))
        assert await connection.run_sync(lambda sync: set(inspect(sync).get_table_names())) == {
            "alembic_version"
        }
        await connection.run_sync(lambda sync: command.upgrade(migration_config(sync), "head"))
        await connection.run_sync(lambda sync: command.check(migration_config(sync)))


async def test_asset_mappings_and_redacted_response(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        assert not camera.is_active
        assert (await CameraRepository(session).set_active(camera.id, True)).is_active
        response = CameraResponse.model_validate(camera).model_dump_json()
        assert "stream_url" not in response and "test-password" not in response
        assert ShelfResponse.model_validate(shelf).roi_version == 1
        assert (await CameraRepository(session).list())[0].id == camera.id


async def test_geometry_cutover_preserves_snapshot_and_rejects_stale_editor(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        repository = ShelfRepository(session)
        old = repository.snapshot(shelf)
        cutover = old.valid_from + timedelta(seconds=1)
        data = GeometryUpdate(
            expected_version=1,
            product_roi={**POLYGON, "reference_revision": uuid4()},
            presence_roi=POLYGON,
            cutover_at=cutover,
        )
        await repository.update_geometry(shelf.id, data)
        response = ShelfResponse.model_validate(shelf)
        assert response.roi_version == 2
        assert response.product_roi.history[0].model_dump(
            exclude={"valid_until"}
        ) == old.model_dump(exclude={"valid_until"})
        assert response.product_roi.history[0].valid_until == cutover
        with pytest.raises(VersionConflictError):
            await repository.update_geometry(shelf.id, data)


async def test_geometry_cutover_refuses_active_interactions(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        await interaction(session, shelf, shelf.created_at)
        with pytest.raises(InvalidReferenceError):
            await ShelfRepository(session).update_geometry(
                shelf.id,
                GeometryUpdate(
                    expected_version=1,
                    product_roi={**POLYGON, "reference_revision": uuid4()},
                    presence_roi=POLYGON,
                    cutover_at=shelf.created_at + timedelta(seconds=2),
                ),
            )


async def test_model_registration_is_idempotent_and_configuration_is_immutable(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        repository = ModelVersionRepository(session)
        data = model_version(ModelKind.POSE)
        registered = await repository.register(data)
        assert (await repository.register(data)).id == registered.id
        changed = data.model_dump(mode="json")
        changed["configuration"]["parameters"]["sampled_fps"] = 15
        with pytest.raises(VersionConflictError):
            await repository.register(type(data).model_validate(changed))
        for statement in (
            update(ModelVersion).where(ModelVersion.id == registered.id).values(version="changed"),
            delete(ModelVersion).where(ModelVersion.id == registered.id),
        ):
            with pytest.raises(IntegrityError):
                async with session.begin_nested():
                    await session.execute(statement)


async def test_jsonb_observation_roundtrip_nullable_targets_and_idempotency(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifiers = await versions(session)
        data = observation(ShelfRepository.snapshot(shelf), identifiers, shelf.created_at)
        repository = ObservationRepository(session)
        saved = await repository.add(data)
        assert saved.attention_shelf_id is None and saved.reach_shelf_id is None
        assert (await repository.add(data)).id == saved.id
        await session.refresh(saved)
        assert ObservationCreate.model_validate(saved) == data
        assert await session.scalar(select(func.count()).select_from(FrameObservation)) == 1
        changed = data.model_dump()
        changed["attributes"]["provenance"]["sampled_fps"] = 12
        with pytest.raises(VersionConflictError):
            await repository.add(ObservationCreate.model_validate(changed))


async def test_typed_link_triple_key_and_duplicate_rejection(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifiers = await versions(session)
        instant = shelf.created_at
        parent = await interaction(session, shelf, instant)
        data = observation(ShelfRepository.snapshot(shelf), identifiers, instant).model_dump()
        data["attention_shelf_id"] = data["reach_shelf_id"] = shelf.id
        observation_row = await ObservationRepository(session).add(
            ObservationCreate.model_validate(data)
        )
        repository = ObservationRepository(session)
        for kind in LinkType:
            await repository.link(observation_row.id, parent.id, kind)
        assert await session.scalar(select(func.count()).select_from(ObservationInteraction)) == 3
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                session.add(
                    ObservationInteraction(
                        observation_id=observation_row.id,
                        interaction_id=parent.id,
                        link_type="PRESENCE",
                    )
                )
                await session.flush()


async def test_cross_camera_observation_is_rejected_in_database(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        other_camera, _ = await assets(session)
        identifiers = await versions(session)
        data = observation(ShelfRepository.snapshot(shelf), identifiers, shelf.created_at)
        values = data.model_dump()
        values.update(camera_id=other_camera.id, attention_shelf_id=shelf.id)
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                session.add(FrameObservation(**values))
                await session.flush()


async def test_business_events_validate_models_evidence_and_jsonb(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifiers = await versions(session)
        instant = shelf.created_at
        parent = await interaction(session, shelf, instant)
        repository = EventRepository(session)
        episode = await repository.add_attention(
            AttentionCreate(
                id=uuid4(),
                interaction_id=parent.id,
                model_id=identifiers[ModelKind.ATTENTION],
                started_at=instant,
                ended_at=None,
                method="head-cone",
                confidence=0.8,
            )
        )
        assert episode.ended_at is None
        action = ActionCreate.model_validate(
            {
                **action_data(),
                "interaction_id": parent.id,
                "model_id": identifiers[ModelKind.ACTION],
                "occurred_at": instant,
            }
        )
        event = await repository.add_action(action)
        await session.refresh(event)
        assert event.attributes["object"]["sku"] is None
        reaction = ReactionCreate.model_validate(
            {
                "id": uuid4(),
                "interaction_id": parent.id,
                "model_id": identifiers[ModelKind.BODY_REACTION],
                "signal_type": "HESITATION",
                "source": "BODY_BIOMECHANICS",
                "occurred_at": instant,
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
                    "temporal_evidence": {"start_frame": 0, "end_frame": 14, "duration_ms": 1400},
                    "association": {"single_shelf_context": True, "context_quality": 0.8},
                    "inferred_emotion": None,
                },
            }
        )
        signal = await repository.add_reaction(reaction)
        await session.refresh(signal)
        assert signal.attributes == reaction.attributes.model_dump(mode="json")
        fused = reaction.model_dump(mode="json")
        fused.update(
            id=str(uuid4()),
            source="FACE_AND_BODY",
            model_id=str(identifiers[ModelKind.FACE_BODY_REACTION]),
        )
        fused["attributes"]["facial_analysis"].update(
            face_visible=True, detected_expression="smile", visibility_quality=0.9
        )
        assert (
            await repository.add_reaction(ReactionCreate.model_validate(fused))
        ).source == "FACE_AND_BODY"
        with pytest.raises(InvalidReferenceError):
            await repository.add_reaction(
                reaction.model_copy(update={"id": uuid4(), "model_id": identifiers[ModelKind.FACE]})
            )
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(
                    update(ReactionSignal)
                    .where(ReactionSignal.id == signal.id)
                    .values(model_id=identifiers[ModelKind.FACE])
                )


async def test_parent_and_children_rollback_and_restrict_deletion(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        identifiers = await versions(session)
        with pytest.raises(RuntimeError):
            async with session.begin_nested():
                parent = await interaction(session, shelf, shelf.created_at)
                data = observation(ShelfRepository.snapshot(shelf), identifiers, shelf.created_at)
                row = await ObservationRepository(session).add(data)
                await ObservationRepository(session).link(row.id, parent.id, LinkType.PRESENCE)
                raise RuntimeError("Simulated failure after child insertion")
        for model in (ShelfInteraction, FrameObservation, ObservationInteraction):
            assert await session.scalar(select(func.count()).select_from(model)) == 0
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(delete(Camera).where(Camera.id == camera.id))


async def test_invalid_durations_metrics_and_missing_parent_are_rejected(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        for row in (
            ShelfMetric(
                shelf_id=shelf.id,
                granularity="hour",
                window_start=shelf.created_at,
                window_end=shelf.created_at + timedelta(hours=1),
                pickup_count=-1,
            ),
            ShelfSignalMetric(
                shelf_id=shelf.id,
                granularity="hour",
                window_start=shelf.created_at,
                signal_type="HESITATION",
                signal_count=-1,
            ),
            ObservationInteraction(
                observation_id=uuid4(), interaction_id=uuid4(), link_type="PRESENCE"
            ),
        ):
            with pytest.raises(IntegrityError):
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()


async def test_retained_history_fails_closed_and_extends_to_active_start(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        instant = shelf.created_at
        await interaction(session, shelf, instant)
        window = TimeWindow(
            start=instant + timedelta(seconds=5), end=instant + timedelta(seconds=10)
        )
        repository = HistoryRepository(session)
        assert not (
            await repository.assess(camera.id, window, HistoryPurpose.UNIQUE_TRACKS, None)
        ).eligible
        coverage = VerifiedCoverage(
            camera_id=camera.id,
            source="RAW",
            start=window.start,
            end=window.end,
            verified_at=window.end,
            linked_observations_complete=True,
        )
        assert (
            await repository.assess(camera.id, window, HistoryPurpose.UNIQUE_TRACKS, coverage)
        ).eligible
        replay = await repository.assess(camera.id, window, HistoryPurpose.REPLAY, coverage)
        assert not replay.eligible and replay.required_start == instant
        assert not (
            await repository.assess(camera.id, window, HistoryPurpose.REAGGREGATION, coverage)
        ).eligible


async def test_concurrent_geometry_editors_preserve_one_cutover(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifier, instant = shelf.id, shelf.created_at

    async def edit() -> str:
        try:
            async with domain_database.transaction() as session:
                await ShelfRepository(session).update_geometry(
                    identifier,
                    GeometryUpdate(
                        expected_version=1,
                        product_roi={**POLYGON, "reference_revision": uuid4()},
                        presence_roi=POLYGON,
                        cutover_at=instant + timedelta(seconds=1),
                    ),
                )
            return "accepted"
        except VersionConflictError:
            return "conflict"

    assert sorted(await asyncio.gather(edit(), edit())) == ["accepted", "conflict"]
    async with domain_database.transaction() as session:
        current = await ShelfRepository(session).get(identifier)
        assert current.roi_version == 2 and len(current.product_roi["history"]) == 1


async def test_delayed_observation_uses_retained_geometry_but_not_expired_interval(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifiers = await versions(session)
        old = ShelfRepository.snapshot(shelf)
        cutover = old.valid_from + timedelta(seconds=2)
        await ShelfRepository(session).update_geometry(
            shelf.id,
            GeometryUpdate(
                expected_version=1,
                product_roi={**POLYGON, "reference_revision": uuid4()},
                presence_roi=POLYGON,
                cutover_at=cutover,
            ),
        )
        repository = ObservationRepository(session)
        saved = await repository.add(observation(old, identifiers, old.valid_from))
        assert saved.attributes["geometry"][0]["roi_version"] == 1
        with pytest.raises(InvalidReferenceError):
            await repository.add(observation(old, identifiers, cutover))


async def test_raw_deletion_order_preserves_business_records(domain_database: Database) -> None:
    async with domain_database.transaction() as session:
        camera, shelf = await assets(session)
        identifiers = await versions(session)
        instant = shelf.created_at
        parent = await interaction(session, shelf, instant, active=False)
        row = await ObservationRepository(session).add(
            observation(ShelfRepository.snapshot(shelf), identifiers, instant)
        )
        await ObservationRepository(session).link(row.id, parent.id, LinkType.PRESENCE)
        parent_id, row_id = parent.id, row.id
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(delete(FrameObservation).where(FrameObservation.id == row_id))
        await session.execute(
            delete(ObservationInteraction).where(ObservationInteraction.observation_id == row_id)
        )
        await session.execute(delete(FrameObservation).where(FrameObservation.id == row_id))
        assert await session.get(ShelfInteraction, parent_id) is not None
        assert await session.scalar(select(func.count()).select_from(ModelVersion)) == len(
            ModelKind
        )
        window = TimeWindow(start=instant, end=instant + timedelta(seconds=1))
        assert not (
            await HistoryRepository(session).assess(
                camera.id, window, HistoryPurpose.UNIQUE_TRACKS, None
            )
        ).eligible


async def test_database_rejects_geometry_history_rewrite_and_negative_presence(
    domain_database: Database,
) -> None:
    async with domain_database.transaction() as session:
        _, shelf = await assets(session)
        identifier, instant = shelf.id, shelf.created_at
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                await session.execute(
                    update(Shelf).where(Shelf.id == identifier).values(roi_version=2)
                )
        with pytest.raises(IntegrityError):
            async with session.begin_nested():
                session.add(
                    ShelfInteraction(
                        id=uuid4(),
                        shelf_id=identifier,
                        started_at=instant,
                        ended_at=None,
                        last_presence_at=instant,
                        presence_duration_ms=-1,
                        status="ACTIVE",
                        end_reason=None,
                        roi_version=1,
                    )
                )
                await session.flush()
