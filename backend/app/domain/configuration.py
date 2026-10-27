"""Stable revisions for operational configuration and source geometry fencing."""

from hashlib import sha256
from uuid import UUID, uuid5

from app.models.camera import Camera

CONFIGURATION_NAMESPACE = UUID("2dab5c12-6e5d-4948-8889-f8439ae3cd84")


def camera_revision(camera: Camera) -> UUID:
    return uuid5(
        CONFIGURATION_NAMESPACE, f"configuration:{camera.id}:{camera.updated_at.isoformat()}"
    )


def source_revision(camera: Camera) -> UUID:
    fingerprint = sha256(camera.stream_url.encode()).hexdigest()
    return uuid5(CONFIGURATION_NAMESPACE, f"source:{camera.id}:{fingerprint}")


def proxy_path(identifier: UUID) -> str:
    return f"cameras/{identifier}"
