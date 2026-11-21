"""Prompt direction: turn a short idea into the dense prompt video models expect.

Seedance pairs its video model with a fine-tuned LLM that rewrites user
prompts into the long, detailed captions the model was trained on. openreel
offers two levels of the same idea:

* ``compose_prompt`` adds camera, style and quality language from fixed
  presets. No model, instant, deterministic.
* ``LLMRewriter`` asks an open instruction-tuned LLM (Qwen2.5 by default) to
  rewrite the idea as a full shot description, then frees it before the video
  model loads.
"""

from __future__ import annotations

import re
from typing import Any

CAMERA_MOVES: dict[str, str] = {
    "static": "Static camera, locked-off tripod shot.",
    "pan-left": "The camera slowly pans to the left.",
    "pan-right": "The camera slowly pans to the right.",
    "tilt-up": "The camera slowly tilts up.",
    "tilt-down": "The camera slowly tilts down.",
    "dolly-in": "The camera slowly dollies in toward the subject.",
    "dolly-out": "The camera slowly pulls back, revealing more of the surroundings.",
    "zoom-in": "Slow, smooth zoom in on the subject.",
    "orbit": "The camera orbits around the subject in a smooth arc.",
    "tracking": "Tracking shot: the camera follows the subject as it moves.",
    "crane-up": "The camera cranes upward, rising above the scene.",
    "handheld": "Handheld camera with subtle, natural shake.",
    "aerial": "Aerial drone shot gliding high over the scene.",
}

STYLES: dict[str, str] = {
    "cinematic": (
        "Cinematic look, shot on a 35mm anamorphic lens, shallow depth of field, "
        "natural film grain, professional color grading."
    ),
    "realistic": "Photorealistic, true-to-life textures and natural motion.",
    "documentary": "Documentary footage, available light, realistic textures, observational framing.",
    "anime": "Japanese anime style, cel shading, clean line art, vibrant colors, expressive animation.",
    "3d-animation": "Feature-film 3D animation style, soft global illumination, appealing stylized characters.",
    "commercial": "High-end commercial, studio lighting, crisp detail, polished product aesthetics.",
    "vintage": "Vintage 16mm film, warm faded colors, light leaks, gentle gate weave.",
    "noir": "Black-and-white film noir, high-contrast chiaroscuro lighting, deep shadows.",
}

QUALITY_SUFFIX = "High detail, smooth natural motion, coherent anatomy."

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery, flickering, distorted, deformed, disfigured, "
    "extra fingers, poorly drawn hands, poorly drawn face, fused fingers, extra limbs, "
    "static frame, still picture, watermark, subtitles, text, JPEG artifacts, messy background"
)


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text[-1] in ".!?" else text + "."


def _lookup(table: dict[str, str], key: str, kind: str) -> str:
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown {kind} {key!r}; choose one of: {', '.join(table)}") from None


def compose_prompt(
    prompt: str,
    *,
    style: str | None = None,
    camera: str | None = None,
    extra: str | None = None,
    quality: bool = True,
) -> str:
    """Expand a prompt with camera, extra details, style and quality language."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    parts = [_sentence(prompt)]
    if camera:
        parts.append(_lookup(CAMERA_MOVES, camera, "camera move"))
    if extra:
        parts.append(_sentence(extra))
    if style:
        parts.append(_lookup(STYLES, style, "style"))
    if quality:
        parts.append(QUALITY_SUFFIX)
    return " ".join(p for p in parts if p)


def apply_cast(text: str, cast: dict[str, str]) -> str:
    """Spell out each named character at its first mention.

    Video models do not know who "Mira" is, so ``Mira walks in`` becomes
    ``Mira (a young woman with silver hair, red raincoat) walks in``. Repeating
    the exact same description in every shot is what keeps a character's look
    stable across separately generated shots. Names match case-sensitively so
    a character called "Rose" does not capture the flower.
    """
    for name, description in cast.items():
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        text = pattern.sub(f"{name} ({description.strip()})", text, count=1)
    return text


SYSTEM_PROMPT = (
    "You rewrite short video ideas into one detailed prompt for a text-to-video model. "
    "Write a single present-tense paragraph of 70 to 120 words that covers: the main subject "
    "and its appearance, the action as it unfolds over a few seconds, the setting, the shot "
    "framing and camera movement, the lighting, and the visual style. Keep every concrete detail "
    "the user gave, including names, colors, counts, text in parentheses and camera directions. "
    "Do not add dialogue, on-screen text or new main characters. Output only the prompt."
)

IMAGE_SYSTEM_SUFFIX = (
    " The video starts from an image the user already has, so focus on motion, action and camera "
    "movement rather than describing the image's contents from scratch."
)

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"


class LLMRewriter:
    """Rewrite prompts with a local open LLM through transformers.

    ``model`` and ``tokenizer`` can be passed in directly (used by the tests);
    otherwise ``model_id`` is loaded on first use.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_LLM,
        *,
        max_new_tokens: int = 220,
        device_map: str = "auto",
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.model = model
        self.tokenizer = tokenizer

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype="auto", device_map=self.device_map
        )

    def messages(self, prompt: str, *, image_to_video: bool = False) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT + (IMAGE_SYSTEM_SUFFIX if image_to_video else "")
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt.strip()}]

    def rewrite(self, prompt: str, *, image_to_video: bool = False) -> str:
        import torch

        self._ensure_loaded()
        inputs = self.tokenizer.apply_chat_template(
            self.messages(prompt, image_to_video=image_to_video),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text or prompt

    def unload(self) -> None:
        """Free the LLM so the video model gets the GPU memory."""
        self.model = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
