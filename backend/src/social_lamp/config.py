from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
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
    enable_cloud_conversation: bool = Field(
        default=False, validation_alias="ENABLE_CLOUD_CONVERSATION"
    )
