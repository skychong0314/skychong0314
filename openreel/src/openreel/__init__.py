"""openreel: Seedance-style video generation with open-weight models."""

from .generator import GenerationRequest, GenerationResult, VideoGenerator
from .models import DEFAULT_MODEL, MODELS, ModelSpec, get_model
from .prompting import LLMRewriter, apply_cast, compose_prompt
from .storyboard import Shot, Storyboard, plan_storyboard, render_storyboard

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "GenerationRequest",
    "GenerationResult",
    "LLMRewriter",
    "ModelSpec",
    "Shot",
    "Storyboard",
    "VideoGenerator",
    "apply_cast",
    "compose_prompt",
    "get_model",
    "plan_storyboard",
    "render_storyboard",
]
