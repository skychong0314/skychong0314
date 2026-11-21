"""Registry of open-weight video models that openreel can drive.

Every entry names the Hugging Face repos and diffusers pipeline classes for
text-to-video (T2V) and, where available, image-to-video (I2V), plus the
defaults each model was trained for. Repo IDs and class names follow the
examples shipped inside diffusers itself.

VRAM figures are rough guidance for a single generation at the default
resolution with model CPU offload enabled; real numbers depend on resolution,
frame count, drivers and attention backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    summary: str
    t2v_repo: str | None
    t2v_pipeline: str | None
    i2v_repo: str | None = None
    i2v_pipeline: str | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 81
    fps: int = 16
    steps: int = 50
    guidance: float | None = 5.0
    size_multiple: int = 16
    frame_multiple: int = 4
    dtype: str = "bfloat16"
    vae_fp32: bool = False
    audio: bool = False
    last_frame: bool = False
    vram_gb: int = 0
    notes: str = ""

    @property
    def modes(self) -> tuple[str, ...]:
        modes = []
        if self.t2v_pipeline:
            modes.append("t2v")
        if self.i2v_pipeline:
            modes.append("i2v")
        return tuple(modes)

    def repo_for(self, mode: str) -> str:
        repo = self.t2v_repo if mode == "t2v" else self.i2v_repo
        if repo is None:
            raise ValueError(f"{self.key} has no {mode} variant")
        return repo

    def pipeline_for(self, mode: str) -> str:
        cls = self.t2v_pipeline if mode == "t2v" else self.i2v_pipeline
        if cls is None:
            raise ValueError(f"{self.key} has no {mode} variant")
        return cls

    def snap_size(self, width: int, height: int) -> tuple[int, int]:
        """Round a resolution down to what the model's VAE/patching accepts."""
        m = self.size_multiple
        return max(m, width // m * m), max(m, height // m * m)

    def snap_frames(self, num_frames: int) -> int:
        """Round a frame count to the nearest valid ``k * frame_multiple + 1``."""
        m = self.frame_multiple
        k = max(1, round((num_frames - 1) / m))
        return k * m + 1

    def frames_for(self, seconds: float, fps: int | None = None) -> int:
        fps = fps or self.fps
        return self.snap_frames(max(2, math.ceil(seconds * fps)))


MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    MODELS[spec.key] = spec
    return spec


register(
    ModelSpec(
        key="wan2.1-1.3b",
        name="Wan 2.1 T2V 1.3B",
        family="wan",
        summary="Smallest model; runs on 8 GB consumer GPUs. Text-to-video only.",
        t2v_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        t2v_pipeline="WanPipeline",
        width=832,
        height=480,
        num_frames=81,
        fps=16,
        steps=50,
        guidance=5.0,
        vae_fp32=True,
        vram_gb=8,
    )
)

register(
    ModelSpec(
        key="wan2.2-5b",
        name="Wan 2.2 TI2V 5B",
        family="wan",
        summary="One model for text- and image-to-video at 720p/24fps. Best default for a 24 GB GPU.",
        t2v_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        t2v_pipeline="WanPipeline",
        i2v_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        i2v_pipeline="WanImageToVideoPipeline",
        width=1280,
        height=704,
        num_frames=121,
        fps=24,
        steps=50,
        guidance=5.0,
        size_multiple=32,
        vae_fp32=True,
        vram_gb=24,
    )
)

register(
    ModelSpec(
        key="wan2.2-a14b",
        name="Wan 2.2 A14B (MoE)",
        family="wan",
        summary="Highest-quality Wan: two 14B experts (high-noise layout, low-noise detail).",
        t2v_repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        t2v_pipeline="WanPipeline",
        i2v_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        i2v_pipeline="WanImageToVideoPipeline",
        width=1280,
        height=720,
        num_frames=81,
        fps=16,
        steps=40,
        guidance=4.0,
        vae_fp32=True,
        last_frame=True,
        vram_gb=48,
        notes="80 GB GPU for comfortable 720p; first+last-frame control is experimental on this checkpoint.",
    )
)

register(
    ModelSpec(
        key="ltx-video",
        name="LTX-Video",
        family="ltx",
        summary="Fastest option; near real-time on a single high-end GPU.",
        t2v_repo="Lightricks/LTX-Video",
        t2v_pipeline="LTXPipeline",
        i2v_repo="Lightricks/LTX-Video",
        i2v_pipeline="LTXImageToVideoPipeline",
        width=704,
        height=480,
        num_frames=121,
        fps=24,
        steps=50,
        guidance=3.0,
        size_multiple=32,
        frame_multiple=8,
        vram_gb=12,
    )
)

register(
    ModelSpec(
        key="ltx-2",
        name="LTX-2 (video + audio)",
        family="ltx2",
        summary="Generates synchronized audio with the video, like Seedance's newer releases.",
        t2v_repo="Lightricks/LTX-2",
        t2v_pipeline="LTX2Pipeline",
        i2v_repo="Lightricks/LTX-2",
        i2v_pipeline="LTX2ImageToVideoPipeline",
        width=768,
        height=512,
        num_frames=121,
        fps=24,
        steps=30,
        guidance=3.0,
        size_multiple=32,
        frame_multiple=8,
        audio=True,
        vram_gb=48,
        notes="19B parameters; on 24 GB cards use --offload sequential (slow).",
    )
)

register(
    ModelSpec(
        key="hunyuan-1.5",
        name="HunyuanVideo 1.5 (480p)",
        family="hunyuan15",
        summary="8.3B model with strong motion and prompt following; T2V and I2V checkpoints.",
        t2v_repo="hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v",
        t2v_pipeline="HunyuanVideo15Pipeline",
        i2v_repo="hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v",
        i2v_pipeline="HunyuanVideo15ImageToVideoPipeline",
        width=848,
        height=480,
        num_frames=121,
        fps=24,
        steps=50,
        guidance=None,
        dtype="float16",
        vram_gb=24,
        notes="Guidance is set by the checkpoint's own guider; I2V takes its size from the input image.",
    )
)

DEFAULT_MODEL = "wan2.2-5b"


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(MODELS)
        raise ValueError(f"unknown model {key!r}; choose one of: {known}") from None
