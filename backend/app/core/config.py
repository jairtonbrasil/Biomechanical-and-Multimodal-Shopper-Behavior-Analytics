import re
from ipaddress import ip_network
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

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

    app_environment: Literal["local", "test", "production"] = "local"
    api_request_max_bytes: int = Field(default=65536, ge=1024, le=1048576)
    api_rate_limit_per_minute: int = Field(default=240, ge=0, le=10000)
    media_rate_limit_per_minute: int = Field(default=1800, ge=1, le=10000)
    rtsp_allowed_networks: tuple[str, ...] = ()
    rtsp_allowed_ports: tuple[int, ...] = (554, 8554, 322)
    mediamtx_api_url: str = "http://mediamtx:9997"
    mediamtx_hls_url: str = "http://mediamtx:8888"
    mediamtx_api_password: SecretStr = Field(default="", repr=False)
    mediamtx_read_password: SecretStr = Field(default="", repr=False)
    mediamtx_publish_password: SecretStr = Field(default="", repr=False)
    mediamtx_rtsp_host: str = "mediamtx"
    mediamtx_rtsp_port: int = Field(default=8554, ge=1, le=65535)
    configuration_poll_seconds: float = Field(default=2, ge=0.1, le=60)
    reference_directory: Path = Path("/var/lib/shopper/references")
    reference_timeout_seconds: float = Field(default=10, ge=1, le=30)
    reference_job_ttl_seconds: int = Field(default=60, ge=10, le=300)
    reference_max_bytes: int = Field(default=4194304, ge=1024, le=16777216)
    reference_storage_limit_bytes: int = Field(default=536870912, ge=1048576)
    reference_queue_limit: int = Field(default=32, ge=1, le=256)
    reference_max_width: int = Field(default=1920, ge=320, le=1920)
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
    vision_camera_id: str = ""
    vision_frame_width: int = Field(default=1280, ge=320, le=1920)
    vision_frame_height: int = Field(default=720, ge=240, le=1920)
    vision_capture_timeout_seconds: float = Field(default=10, ge=1, le=30)
    vision_reconnect_max_seconds: float = Field(default=10, ge=0.1, le=60)
    vision_reconnect_attempts: int = Field(default=5, ge=1, le=100)
    vision_confidence: float = Field(default=0.3, gt=0, le=1)
    vision_image_size: int = Field(default=640, ge=320, le=1280)
    vision_max_people: int = Field(default=32, ge=1, le=32)
    vision_device: str = Field(default="cpu", pattern=r"^(cpu|[0-9])$")
    vision_cpu_threads: int = Field(default=2, ge=1, le=32)
    vision_max_frames: int = Field(default=300, ge=0, le=1000000)
    vision_max_seconds: float = Field(default=120, ge=1, le=86400)
    vision_summary_seconds: float = Field(default=5, ge=1, le=60)
    vision_publish_timeout_seconds: float = Field(default=5, ge=0.1, le=30)
    vision_publish_attempts: int = Field(default=3, ge=1, le=10)
    vision_output_directory: Path = Path("/outputs/stage4")
    vision_preview_enabled: bool = True
    vision_preview_max_seconds: int = Field(default=60, ge=1, le=600)
    vision_observation_sample_limit: int = Field(default=50, ge=0, le=1000)
    vision_model_manifest: Path = Path("/models/manifest.json")
    pose_model_path: Path = Path("/models/yolo11s-pose.pt")
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

    @model_validator(mode="after")
    def validate_operational_settings(self) -> Self:
        networks = [ip_network(value, strict=True) for value in self.rtsp_allowed_networks]
        origin = urlsplit(self.mediamtx_api_url)
        if (
            origin.scheme not in {"http", "https"}
            or not origin.hostname
            or origin.username is not None
            or origin.password is not None
            or origin.query
            or origin.fragment
            or origin.path not in {"", "/"}
        ):
            raise ValueError("Media control must use an explicit HTTP(S) origin")
        if not self.rtsp_allowed_ports or any(
            not 1 <= port <= 65535 for port in self.rtsp_allowed_ports
        ):
            raise ValueError("RTSP ports must be nonempty valid port numbers")
        for secret in (
            self.mediamtx_api_password,
            self.mediamtx_read_password,
            self.mediamtx_publish_password,
        ):
            value = secret.get_secret_value()
            if value and not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", value):
                raise ValueError("Media credentials must use 32–128 URL-safe characters")
        if self.reference_job_ttl_seconds <= self.reference_timeout_seconds + 5:
            raise ValueError("Reference job lifetime must exceed its capture deadline")
        if self.app_environment == "production":
            if not networks or not self.api_rate_limit_per_minute:
                raise ValueError("Production requires source allowlists and quotas")
            forbidden = (ip_network("127.0.0.0/8"), ip_network("::1/128"))
            if any(
                network.prefixlen == 0
                or any(
                    network.version == other.version and network.overlaps(other)
                    for other in forbidden
                )
                for network in networks
            ):
                raise ValueError("Production source networks must be explicit non-loopback ranges")
            if not self.mediamtx_api_password.get_secret_value():
                raise ValueError("Production requires authenticated media control")
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
