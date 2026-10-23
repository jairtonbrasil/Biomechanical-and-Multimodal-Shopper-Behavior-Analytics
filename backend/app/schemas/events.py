from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.schemas.common import (
    ActionType,
    CloseReason,
    Contract,
    NonNegative,
    ReactionSource,
    UnitFloat,
    require_ordered,
)


class TemporalEvidence(Contract):
    start_frame: NonNegative
    end_frame: NonNegative
    duration_ms: NonNegative

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_frame < self.start_frame:
            raise ValueError("Evidence frame interval is reversed")
        return self


class ObjectEvidence(Contract):
    detected: bool | None
    class_name: str | None = Field(alias="class", max_length=128)
    product_id: str | None = Field(max_length=128)
    sku: str | None = Field(max_length=128)
    tracking_confidence: UnitFloat | None


class InteractionEvidence(Contract):
    hand_entered_roi: bool | None
    object_displacement_detected: bool | None
    object_left_shelf: bool | None


class ActionAttributes(Contract):
    hand: Literal["left", "right"]
    detection_method: str = Field(min_length=1, max_length=128)
    object: ObjectEvidence
    interaction: InteractionEvidence
    temporal_evidence: TemporalEvidence
    evidence_provider: str | None = Field(max_length=128)
    bimanual: bool


class ActionCreate(Contract):
    id: UUID
    interaction_id: UUID
    model_id: UUID
    action_type: ActionType
    occurred_at: AwareDatetime
    confidence: UnitFloat
    attributes: ActionAttributes

    @model_validator(mode="after")
    def supported_product_action(self) -> Self:
        if self.action_type in {ActionType.PICKUP, ActionType.RETURN}:
            evidence = self.attributes
            if (
                not evidence.evidence_provider
                or evidence.object.detected is not True
                or evidence.interaction.object_displacement_detected is not True
                or evidence.interaction.object_left_shelf
                is not (self.action_type == ActionType.PICKUP)
            ):
                raise ValueError(
                    "Pickup/return requires provider-backed object displacement evidence"
                )
        return self


class FacialAnalysis(Contract):
    face_visible: bool | None
    detected_expression: str | None = Field(max_length=64)
    visibility_quality: UnitFloat | None
    occlusion_estimated: UnitFloat | None

    @model_validator(mode="after")
    def observed_expression(self) -> Self:
        if self.face_visible is not True and self.detected_expression is not None:
            raise ValueError("Facial labels require a visible face")
        return self


class BehavioralContext(Contract):
    person_standing: bool | None
    attention_detected: bool | None
    handling_product: bool | None


class Association(Contract):
    single_shelf_context: bool | None
    context_quality: UnitFloat | None


class ReactionAttributes(Contract):
    facial_analysis: FacialAnalysis
    behavioral_context: BehavioralContext
    temporal_evidence: TemporalEvidence
    association: Association
    inferred_emotion: str | None = Field(max_length=64)


class ReactionCreate(Contract):
    id: UUID
    interaction_id: UUID
    model_id: UUID
    signal_type: str = Field(min_length=1, max_length=64)
    source: ReactionSource
    occurred_at: AwareDatetime
    confidence: UnitFloat
    attributes: ReactionAttributes

    @model_validator(mode="after")
    def source_matches_evidence(self) -> Self:
        face = self.attributes.facial_analysis
        if self.source == ReactionSource.FACE_AND_BODY:
            if face.face_visible is not True or face.detected_expression is None:
                raise ValueError("Face/body fusion requires observed facial evidence")
        elif face.detected_expression is not None or self.attributes.inferred_emotion is not None:
            raise ValueError("Body-only signals cannot manufacture facial labels or emotions")
        return self


class InteractionCreate(Contract):
    id: UUID
    shelf_id: UUID
    started_at: AwareDatetime
    ended_at: AwareDatetime | None
    last_presence_at: AwareDatetime
    presence_duration_ms: NonNegative
    status: Literal["ACTIVE", "CLOSED"]
    end_reason: CloseReason | None
    roi_version: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_lifecycle(self) -> Self:
        require_ordered(self.started_at, self.ended_at)
        if self.last_presence_at < self.started_at or (
            self.ended_at is not None and self.last_presence_at > self.ended_at
        ):
            raise ValueError("Presence endpoint lies outside the interaction interval")
        if self.status == "ACTIVE" and (self.ended_at is not None or self.end_reason is not None):
            raise ValueError("Active interactions have no end or closure reason")
        if self.status == "CLOSED" and (self.ended_at is None or self.end_reason is None):
            raise ValueError("Closed interactions require an end and closure reason")
        return self


class AttentionCreate(Contract):
    id: UUID
    interaction_id: UUID
    model_id: UUID
    started_at: AwareDatetime
    ended_at: AwareDatetime | None
    method: str = Field(min_length=1, max_length=64)
    confidence: UnitFloat

    @model_validator(mode="after")
    def ordered(self) -> Self:
        require_ordered(self.started_at, self.ended_at)
        return self
