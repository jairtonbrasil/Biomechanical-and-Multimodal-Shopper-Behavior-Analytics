from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

ROOT_DIRECTORY = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIRECTORY / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        frozen=True,
    )

    app_environment: Literal["local", "test"] = "local"
    database_host: str = "postgres"
    database_port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = "shopper_analytics"
    database_user: str = "shopper"
    database_password: SecretStr = Field(repr=False)
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=5, ge=0, le=50)
    redis_host: str = "redis"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_password: SecretStr = Field(repr=False)
    redis_max_connections: int = Field(default=20, ge=1, le=100)
    redis_namespace: str = Field(default="shopper", pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    cache_ttl_seconds: int = Field(default=120, ge=60, le=300)
    state_ttl_seconds: int = Field(default=60, ge=10, le=86400)
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_observations_topic: str = "retail.frame-observations"
    kafka_consumer_group: str = "retail.temporal-processor"
    kafka_max_message_bytes: int = Field(default=262144, ge=1024, le=1048576)
    kafka_max_poll_records: int = Field(default=100, ge=1, le=1000)
    kafka_max_poll_interval_ms: int = Field(default=300000, ge=10000, le=900000)
    dependency_timeout_seconds: float = Field(default=5, ge=0.1, le=30)
    kafka_request_timeout_ms: int = Field(default=10000, ge=1000, le=30000)
    entry_confirmation_ms: int = Field(default=500, gt=0)
    exit_tolerance_ms: int = Field(default=2000, gt=0)
    tracking_loss_tolerance_ms: int = Field(default=5000, gt=0)
    sampling_fps: float = Field(default=10, gt=0, le=60)
    pose_model_path: Path = Path("/models/yolo11n-pose.pt")
    face_model_path: Path = Path("/models/hsemotion.onnx")

    @field_validator("database_password", "redis_password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 16 or value.get_secret_value().isspace():
            raise ValueError("A password of at least 16 characters is required")
        return value

    @model_validator(mode="after")
    def validate_state_lifetime(self) -> Self:
        if self.state_ttl_seconds * 1000 <= self.tracking_loss_tolerance_ms:
            raise ValueError("State TTL must exceed the tracking loss tolerance")
        return self

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=self.database_user,
            password=self.database_password.get_secret_value(),
            host=self.database_host,
            port=self.database_port,
            database=self.database_name,
        )
