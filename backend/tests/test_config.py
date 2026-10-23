import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize("lifetime", [0, 59, 301])
def test_invalid_cache_lifetime_is_rejected(lifetime: int) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_password="database-test-password",
            redis_password="redis-test-password",
            cache_ttl_seconds=lifetime,
        )


@pytest.mark.parametrize("name", ["DATABASE_PASSWORD", "REDIS_PASSWORD"])
def test_credentials_are_required(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.delenv(name, raising=False)
    other = "redis_password" if name == "DATABASE_PASSWORD" else "database_password"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{other: "independent-test-password"})


def test_settings_and_validation_representations_hide_credentials(settings: Settings) -> None:
    for value in (str(settings), repr(settings), settings.model_dump_json()):
        assert "database-test-password" not in value
        assert "redis-test-password" not in value
    assert "database-test-password" not in str(settings.database_url)
    with pytest.raises(ValidationError) as result:
        Settings(_env_file=None, database_password="short-secret", redis_password="also-short")
    assert "short-secret" not in str(result.value)


def test_state_lifetime_exceeds_loss_tolerance() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_password="database-test-password",
            redis_password="redis-test-password",
            state_ttl_seconds=10,
            tracking_loss_tolerance_ms=10000,
        )
