from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_version import ModelVersion
from app.repositories.errors import InvalidReferenceError, VersionConflictError
from app.schemas.common import ModelKind
from app.schemas.model_version import ModelVersionCreate


class ModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def require_kind(self, identifier: UUID, expected: ModelKind) -> ModelVersion:
        model = await self.session.get(ModelVersion, identifier)
        if model is None or model.kind != expected:
            raise InvalidReferenceError("Model version is absent or has an incompatible kind")
        return model

    async def register(self, data: ModelVersionCreate) -> ModelVersion:
        for kind, identifier in data.configuration.dependencies.items():
            await self.require_kind(identifier, kind)
        configuration = data.configuration.model_dump(mode="json")
        await self.session.execute(
            insert(ModelVersion)
            .values(
                id=uuid4(),
                name=data.name,
                kind=data.kind,
                version=data.version,
                configuration=configuration,
            )
            .on_conflict_do_nothing(index_elements=["name", "kind", "version"])
        )
        model = await self.session.scalar(
            select(ModelVersion).where(
                ModelVersion.name == data.name,
                ModelVersion.kind == data.kind,
                ModelVersion.version == data.version,
            )
        )
        if model is None:
            raise RuntimeError("Registered model version could not be retrieved")
        if model.configuration != configuration:
            raise VersionConflictError("Register a new version for a changed model configuration")
        return model
