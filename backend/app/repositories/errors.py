class EntityNotFoundError(ValueError):
    """A required parent or version is absent."""


class VersionConflictError(ValueError):
    """Immutable configuration or geometry conflicts with the expected version."""


class InvalidReferenceError(ValueError):
    """A reference violates camera, geometry, model or evidence scope."""
