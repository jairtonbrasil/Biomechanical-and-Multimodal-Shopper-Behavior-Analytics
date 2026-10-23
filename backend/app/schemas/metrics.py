from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.schemas.common import Contract, NonNegative


class MetricKey(Contract):
    shelf_id: UUID
    granularity: str = Field(min_length=1, max_length=32)
    window_start: AwareDatetime


class ShelfMetrics(MetricKey):
    window_end: AwareDatetime
    interactions_started: NonNegative
    interactions_with_attention: NonNegative
    attention_episodes: NonNegative
    presence_duration_ms: NonNegative
    attention_duration_ms: NonNegative
    reach_count: NonNegative
    touch_count: NonNegative
    pickup_count: NonNegative
    return_count: NonNegative

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.window_end <= self.window_start:
            raise ValueError("Metric windows are half-open and require a positive duration")
        return self


class SignalMetrics(MetricKey):
    signal_type: str = Field(min_length=1, max_length=64)
    signal_count: NonNegative
