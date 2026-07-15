from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        secrets_dir=None,
    )

    conversation_provider: str = Field(default="template", validation_alias="CONVERSATION_PROVIDER")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_realtime_model: str = Field(
        default="gpt-4o-realtime-preview", validation_alias="OPENAI_REALTIME_MODEL"
    )
    database_path: Path = Field(
        default=Path(".runtime/memory.db"), validation_alias="DATABASE_PATH"
    )
    snapshot_path: Path = Field(
        default=Path(".runtime/snapshots"), validation_alias="SNAPSHOT_PATH"
    )
    camera_index: int = Field(default=0, validation_alias="CAMERA_INDEX")
    retention_days: int = Field(default=30, validation_alias="RETENTION_DAYS")
    enable_audio_bonus: bool = Field(default=False, validation_alias="ENABLE_AUDIO_BONUS")
    enable_live_capture: bool = Field(default=False, validation_alias="ENABLE_LIVE_CAPTURE")
    enable_mediapipe_face_landmarker: bool = Field(
        default=False, validation_alias="ENABLE_MEDIAPIPE_FACE_LANDMARKER"
    )
    enable_cloud_conversation: bool = Field(
        default=False, validation_alias="ENABLE_CLOUD_CONVERSATION"
    )
    enable_object_detection: bool = Field(
        default=False, validation_alias="ENABLE_OBJECT_DETECTION"
    )
    object_detector_model: str = Field(
        default="yolov8n.pt", validation_alias="OBJECT_DETECTOR_MODEL"
    )
    object_detection_confidence: float = Field(
        default=0.45, validation_alias="OBJECT_DETECTION_CONFIDENCE"
    )
    object_detection_max_fps: int = Field(
        default=8, validation_alias="OBJECT_DETECTION_MAX_FPS"
    )
    object_detection_classes: str | None = Field(
        default=None, validation_alias="OBJECT_DETECTION_CLASSES"
    )
