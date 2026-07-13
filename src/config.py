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
    # Reserve ~60 s for image-pull + container startup + output write under
    # the 600 s hard cap.  Was 570 (9.5 min); tightened to 540 (9 min) so
    # a slow image pull still leaves headroom.
    max_runtime_seconds: int = 540
    frames_min: int = 8
    # 16 frames gives strong VLM coverage while cutting per-task payload vs
    # the old 20-frame default (~20 % faster grounding TTFT on large clips).
    frames_max: int = 16

    grounding_model: str = "accounts/fireworks/models/kimi-k2p6"
    styling_model: str = "accounts/fireworks/models/deepseek-v4-pro"
    judge_model: str = "accounts/fireworks/models/gpt-oss-120b"

    # Optional secondary models for provider/rate-limit fallback (Phase 6 /
    # Prompt 9).  Defaulted to None so a missing/down secondary provider
    # doesn't silently double LLM calls against the same endpoint.
    secondary_grounding_model: str | None = None
    secondary_styling_model: str | None = None
    secondary_judge_model: str | None = None

    fallback_caption: str = "A short video clip."
    # Capped at 1 (was 2) to halve judge-pass LLM spend: each weak style
    # can now trigger at most 2 total calls (1 judge + 1 regen) instead of 4.
    max_self_judge_retries: int = 1

    # --- Pipeline stage gates -------------------------------------------
    # Chain-of-Verification: 3 sequential LLM calls per task.  Disabled by
    # default (set ENABLE_VERIFICATION=true to re-enable for accuracy runs).
    enable_verification: bool = False
    # Self-judge pass: concurrent per-style judge + conditional regen.
    # Enabled by default; set ENABLE_JUDGE=false to skip entirely when the
    # runtime budget is dangerously tight.
    enable_judge: bool = True


def _parse_bool(value: str) -> bool:
    """Case-insensitive parse of common truthy/falsy strings."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    bool_env = {
        "enable_verification": "ENABLE_VERIFICATION",
        "enable_judge": "ENABLE_JUDGE",
    }
    for field, env_name in bool_env.items():
        value = os.environ.get(env_name)
        if value is not None:
            kwargs[field] = _parse_bool(value)

    try:
        return Settings(**kwargs)
    except ValidationError as exc:
        raise SettingsError(f"Invalid configuration: {exc}") from exc
