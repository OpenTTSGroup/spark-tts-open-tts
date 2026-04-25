from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # --- Engine (SPARKTTS_* prefix) ------------------------------------------
    sparktts_model: str = Field(
        default="SparkAudio/Spark-TTS-0.5B",
        description="HuggingFace repo id or local directory containing "
        "config.yaml + LLM/ + BiCodec/ + wav2vec2-large-xlsr-53/. "
        "If the value is an existing directory it is used directly; "
        "otherwise it is fetched via huggingface_hub.snapshot_download.",
    )
    sparktts_device: Literal["auto", "cuda", "cpu"] = "auto"
    sparktts_cuda_index: int = Field(default=0, ge=0)
    sparktts_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    # Generation defaults (used when the request omits the field)
    sparktts_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    sparktts_top_k: int = Field(default=50, ge=0)
    sparktts_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    sparktts_max_new_tokens: int = Field(default=3000, ge=1, le=16384)

    sparktts_prompt_cache_size: int = Field(default=16, ge=1)

    # Design-path defaults when instruct=null and structured fields omitted
    sparktts_default_gender: Literal["female", "male"] = "female"
    sparktts_default_pitch: Literal[
        "very_low", "low", "moderate", "high", "very_high"
    ] = "moderate"
    sparktts_default_speed: Literal[
        "very_low", "low", "moderate", "high", "very_high"
    ] = "moderate"

    # --- Service-level (no prefix) -------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    voices_dir: str = "/voices"
    max_input_chars: int = Field(default=8000, ge=1)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = "mp3"
    max_concurrency: int = Field(default=1, ge=1)
    max_queue_size: int = Field(default=0, ge=0)
    queue_timeout: float = Field(default=0.0, ge=0.0)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    cors_enabled: bool = False

    @property
    def voices_path(self) -> Path:
        return Path(self.voices_dir)

    @property
    def resolved_device(self) -> str:
        if self.sparktts_device == "cpu":
            return "cpu"
        if self.sparktts_device == "cuda":
            return f"cuda:{self.sparktts_cuda_index}"
        # auto
        import torch

        if torch.cuda.is_available():
            return f"cuda:{self.sparktts_cuda_index}"
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
