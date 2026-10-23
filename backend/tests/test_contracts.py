from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.domain.identity import TrackScope, business_event_id, interaction_id, observation_id
from app.models import Base
from app.schemas.common import ActionType, ModelKind
from app.schemas.events import ActionAttributes, ActionCreate
from app.schemas.geometry import Polygon
from app.schemas.observation import ObservationEnvelope
from tests.factories import POLYGON, example_envelope, model_version


def test_exactly_eleven_domain_tables() -> None:
    assert set(Base.metadata.tables) == {
        "camera",
        "shelf",
        "model_version",
        "shelf_interaction",
        "frame_observation",
        "observation_interaction",
        "attention_episode",
        "action_event",
        "reaction_signal",
        "shelf_metric",
        "shelf_signal_metric",
    }


@pytest.mark.parametrize(
    "points",
    [POLYGON["points"], list(reversed(POLYGON["points"])), [[0, 0], [1, 0], [0.3, 0.3], [0, 1]]],
)
def test_valid_boundary_and_concave_polygons(points: list) -> None:
    assert Polygon.model_validate({**POLYGON, "points": points})


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 1], [1, 0], [0, 1]],
        [[0, 0], [0.5, 0.5], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 0]],
        [[0, 0], [1, 0], [0.5, 0], [0, 1]],
        [[0, 0], [1.1, 0], [0, 1]],
        [[0, 0], [float("nan"), 0], [0, 1]],
        [[0, 0], [float("inf"), 0], [0, 1]],
    ],
)
def test_invalid_geometry_is_rejected(points: list) -> None:
    with pytest.raises(ValidationError):
        Polygon.model_validate({**POLYGON, "points": points})


def test_deterministic_scope_and_delimiters() -> None:
    scope = TrackScope(UUID(int=1), UUID(int=2), "a:b")
    identifier = observation_id(scope, 5)
    assert identifier == UUID("8afa6b5d-d08a-5d7f-84d7-0541a62dab12")
    assert identifier == observation_id(scope, 5)
    assert (
        len(
            {
                identifier,
                observation_id(replace(scope, camera_id=UUID(int=3)), 5),
                observation_id(replace(scope, tracking_session_id=UUID(int=3)), 5),
                observation_id(replace(scope, local_track_id="a"), 5),
                observation_id(scope, 6),
            }
        )
        == 5
    )
    parent = interaction_id(scope, UUID(int=4), 1, 5, UUID(int=5))
    assert parent != interaction_id(scope, UUID(int=4), 2, 5, UUID(int=5))
    assert parent != interaction_id(scope, UUID(int=4), 1, 5, UUID(int=6))
    assert business_event_id(parent, ActionType.REACH, 5, UUID(int=5)) != business_event_id(
        parent, ActionType.INSPECT, 5, UUID(int=5)
    )


def test_required_nulls_round_trip_without_neutral_invention() -> None:
    envelope = example_envelope()
    restored = ObservationEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope
    assert restored.observation.attributes.face.expression_detected is None
    assert restored.observation.attributes.movement.state is None
    payload = envelope.model_dump(mode="json")
    del payload["observation"]["attributes"]["face"]["confidence"]
    with pytest.raises(ValidationError):
        ObservationEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    "case",
    [
        "unsupported_version",
        "missing_group",
        "cross_camera",
        "invented_face",
        "invented_kinematics",
        "target_without_geometry",
        "unknown_field",
    ],
)
def test_observation_contract_rejects_invalid_evidence(case: str) -> None:
    data = example_envelope().model_dump(mode="json")
    observation = data["observation"]
    attributes = observation["attributes"]
    if case == "unsupported_version":
        data["schema_version"] = "1.0"
    elif case == "missing_group":
        del attributes["spatial"]
    elif case == "cross_camera":
        attributes["geometry"][0]["camera_id"] = str(uuid4())
    elif case == "invented_face":
        attributes["face"]["expression_detected"] = "neutral"
    elif case == "invented_kinematics":
        attributes["kinematics"]["reach_candidate"] = True
    elif case == "target_without_geometry":
        observation["attention_shelf_id"] = str(uuid4())
    else:
        attributes["person_identity"] = "unsupported"
    with pytest.raises(ValidationError):
        ObservationEnvelope.model_validate(data)


@pytest.mark.parametrize("kind", [ModelKind.POSE, ModelKind.FACE, ModelKind.TEMPORAL_RULES])
def test_base_model_configurations_are_explicit(kind: ModelKind) -> None:
    assert model_version(kind).kind == kind


def test_model_configuration_rejects_secrets_and_missing_dependencies() -> None:
    data = model_version(ModelKind.POSE).model_dump(mode="json")
    data["configuration"]["parameters"] = {"nested": {"api_key": "do-not-persist"}}
    with pytest.raises(ValidationError):
        type(model_version(ModelKind.POSE)).model_validate(data)
    with pytest.raises(ValidationError):
        model_version(ModelKind.FACE_BODY_REACTION)


def action_data() -> dict:
    return {
        "id": uuid4(),
        "interaction_id": uuid4(),
        "model_id": uuid4(),
        "action_type": "REACH",
        "occurred_at": "2026-01-01T00:00:00Z",
        "confidence": 0.8,
        "attributes": {
            "hand": "left",
            "detection_method": "pose-roi",
            "object": {
                "detected": None,
                "class": None,
                "product_id": None,
                "sku": None,
                "tracking_confidence": None,
            },
            "interaction": {
                "hand_entered_roi": True,
                "object_displacement_detected": None,
                "object_left_shelf": None,
            },
            "temporal_evidence": {"start_frame": 1, "end_frame": 3, "duration_ms": 200},
            "evidence_provider": None,
            "bimanual": False,
        },
    }


def test_action_enum_and_product_evidence() -> None:
    data = action_data()
    assert ActionCreate.model_validate(data).attributes.object.product_id is None
    assert (
        "class"
        in ActionCreate.model_validate(data).attributes.model_dump(mode="json", by_alias=True)[
            "object"
        ]
    )
    for kind in ("TOUCH", "PICKUP", "RETURN"):
        with pytest.raises(ValidationError):
            ActionCreate.model_validate({**data, "action_type": kind})
    with pytest.raises(ValidationError):
        ActionAttributes.model_validate({**data["attributes"], "hand": "both"})
