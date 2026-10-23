import re
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, JsonValue, model_validator

from app.schemas.common import Contract, ModelKind, Name


class ModelConfiguration(Contract):
    schema_version: Literal["1.0"]
    artifact_name: str | None = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    artifact_sha256: str | None = Field(pattern=r"^[a-f0-9]{64}$")
    parameters: dict[str, JsonValue]
    dependencies: dict[ModelKind, UUID]

    @model_validator(mode="after")
    def safe_configuration(self) -> Self:
        def inspect(value: JsonValue) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if re.search(r"password|secret|token|credential|api_key", key, re.I):
                        raise ValueError("Model configuration cannot contain secrets")
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
            elif isinstance(value, str) and ("://" in value or "-----BEGIN" in value):
                raise ValueError("Use artifact identifiers, not URLs or credentials")

        inspect(self.parameters)
        if (self.artifact_name is None) != (self.artifact_sha256 is None):
            raise ValueError("Artifact name and checksum must be supplied together")
        return self


class ModelVersionCreate(Contract):
    name: Name
    kind: ModelKind
    version: Name
    configuration: ModelConfiguration

    @model_validator(mode="after")
    def required_dependencies(self) -> Self:
        required = {
            ModelKind.ATTENTION: {ModelKind.POSE, ModelKind.TEMPORAL_RULES},
            ModelKind.ACTION: {ModelKind.POSE, ModelKind.TEMPORAL_RULES},
            ModelKind.BODY_REACTION: {ModelKind.POSE, ModelKind.TEMPORAL_RULES},
            ModelKind.FACE_BODY_REACTION: {
                ModelKind.POSE,
                ModelKind.FACE,
                ModelKind.TEMPORAL_RULES,
            },
        }.get(self.kind, set())
        if not required.issubset(self.configuration.dependencies):
            raise ValueError("The model configuration is missing required dependency versions")
        if self.kind in {ModelKind.POSE, ModelKind.FACE} and not self.configuration.artifact_sha256:
            raise ValueError("Neural model artifacts require a checksum")
        return self
