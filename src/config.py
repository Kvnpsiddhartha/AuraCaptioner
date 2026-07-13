"""Runtime configuration for the video captioning agent.

`Settings` is imported everywhere; other modules only ever READ its fields,
they do not redeclare them. `load_settings()` is the single place required
env vars are validated, and it fails loudly and early (at process startup,
before any task processing begins) rather than mid-run.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, ValidationError


class SettingsError(Exception):
    """Raised at startup when required configuration is missing or invalid."""


class Settings(BaseModel):
    max_runtime_seconds: int = 570  # 9.5 min, safety margin under the 10:00 hard cap
    frames_min: int = 8
    frames_max: int = 20

    grounding_model: str = "accounts/fireworks/models/kimi-k2p6"
    styling_model: str = "accounts/fireworks/models/deepseek-v4-pro"
    judge_model: str = "accounts/fireworks/models/gpt-oss-120b"

    # Optional secondary models for provider/rate-limit fallback (Phase 6 / Prompt 9).
    secondary_grounding_model: str | None = "accounts/fireworks/models/kimi-k2p6"
    secondary_styling_model: str | None = "accounts/fireworks/models/deepseek-v4-pro"
    secondary_judge_model: str | None = "accounts/fireworks/models/gpt-oss-120b"

    fallback_caption: str = "A short video clip."
    max_self_judge_retries: int = 2


def load_settings() -> Settings:
    """Build Settings from environment variables.

    Uses default values for missing configuration parameters.
    """
    kwargs: dict[str, object] = {}

    mapping = {
        "grounding_model": "GROUNDING_MODEL",
        "styling_model": "STYLING_MODEL",
        "judge_model": "JUDGE_MODEL",
        "secondary_grounding_model": "SECONDARY_GROUNDING_MODEL",
        "secondary_styling_model": "SECONDARY_STYLING_MODEL",
        "secondary_judge_model": "SECONDARY_JUDGE_MODEL",
        "fallback_caption": "FALLBACK_CAPTION",
    }
    for field, env_name in mapping.items():
        value = os.environ.get(env_name)
        if value:
            kwargs[field] = value


    int_env = {
        "max_runtime_seconds": "MAX_RUNTIME_SECONDS",
        "frames_min": "FRAMES_MIN",
        "frames_max": "FRAMES_MAX",
        "max_self_judge_retries": "MAX_SELF_JUDGE_RETRIES",
    }
    for field, env_name in int_env.items():
        value = os.environ.get(env_name)
        if value:
            try:
                kwargs[field] = int(value)
            except ValueError as exc:
                raise SettingsError(
                    f"Environment variable {env_name}={value!r} must be an integer."
                ) from exc

    try:
        return Settings(**kwargs)
    except ValidationError as exc:
        raise SettingsError(f"Invalid configuration: {exc}") from exc

