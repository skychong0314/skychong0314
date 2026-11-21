"""Load open video models through diffusers and run one generation."""

from __future__ import annotations

import gc
import inspect
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .media import fit_image, save_video, to_audio_array, to_float_frames
from .models import ModelSpec

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str | None = None
    image: Image.Image | None = None
    last_image: Image.Image | None = None
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    duration: float | None = None
    fps: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None


@dataclass
class GenerationResult:
    frames: np.ndarray
    fps: int
    seed: int
    prompt: str
    negative_prompt: str | None
    model: str
    mode: str
    width: int
    height: int
    steps: int
    guidance: float | None
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "mode": self.mode,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "duration_s": round(self.duration, 3),
            "steps": self.steps,
            "guidance": self.guidance,
            "audio": self.audio is not None,
            **self.extra,
        }

    def save(self, path: str | Path, *, manifest: bool = True) -> Path:
        path = save_video(path, self.frames, self.fps, audio=self.audio, sample_rate=self.sample_rate)
        if manifest:
            path.with_suffix(".json").write_text(json.dumps(self.metadata(), indent=2) + "\n")
        return path


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def accepted_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split kwargs into those ``fn`` accepts and those it does not.

    Pipelines disagree on parameters (HunyuanVideo 1.5 has no
    ``guidance_scale``, its I2V variant has no ``height``/``width``, LTX wants
    ``frame_rate``), so a superset is built and filtered per pipeline.
    """
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    keep = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(keep))
    return keep, dropped


class VideoGenerator:
    """Runs text-to-video and image-to-video for one :class:`ModelSpec`.

    Pipelines load lazily on first use. When T2V and I2V share a checkpoint
    (Wan 2.2 TI2V, LTX), the second pipeline reuses the first one's weights via
    ``from_pipe`` instead of loading them twice. ``t2v_pipe``/``i2v_pipe`` can be
    supplied directly, which the tests use to run tiny models on CPU.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str | None = None,
        offload: str = "model",
        vae_tiling: bool = True,
        t2v_pipe: Any = None,
        i2v_pipe: Any = None,
    ) -> None:
        if offload not in ("model", "sequential", "none"):
            raise ValueError("offload must be 'model', 'sequential' or 'none'")
        self.spec = spec
        self.device = device
        self.offload = offload
        self.vae_tiling = vae_tiling
        self._pipes: dict[str, Any] = {}
        if t2v_pipe is not None:
            self._pipes["t2v"] = t2v_pipe
        if i2v_pipe is not None:
            self._pipes["i2v"] = i2v_pipe

    # -- loading ---------------------------------------------------------

    def _dtype(self) -> Any:
        import torch

        return getattr(torch, self.spec.dtype)

    def _load(self, mode: str) -> Any:
        import diffusers
        import torch

        spec = self.spec
        cls = getattr(diffusers, spec.pipeline_for(mode))
        repo = spec.repo_for(mode)
        other = "i2v" if mode == "t2v" else "t2v"
        if other in self._pipes and other in spec.modes and spec.repo_for(other) == repo:
            log.info("reusing %s weights for %s via from_pipe", other, mode)
            pipe = cls.from_pipe(self._pipes[other])
            # Offload hooks follow each pipeline's own call order, so re-apply them.
            self._place(pipe)
            return pipe

        log.info("loading %s (%s) from %s", spec.name, mode, repo)
        kwargs: dict[str, Any] = {"torch_dtype": self._dtype()}
        if spec.vae_fp32:
            kwargs["vae"] = diffusers.AutoencoderKLWan.from_pretrained(
                repo, subfolder="vae", torch_dtype=torch.float32
            )
        pipe = cls.from_pretrained(repo, **kwargs)
        self._place(pipe)
        return pipe

    def _place(self, pipe: Any) -> None:
        device = self.device or pick_device()
        if device == "cuda" and self.offload == "model":
            pipe.enable_model_cpu_offload()
        elif device == "cuda" and self.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
        vae = getattr(pipe, "vae", None)
        if self.vae_tiling and vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

    def pipeline(self, mode: str) -> Any:
        if mode not in self.spec.modes:
            raise ValueError(
                f"{self.spec.key} does not support {mode}; it supports: {', '.join(self.spec.modes)}"
            )
        if mode not in self._pipes:
            self._pipes[mode] = self._load(mode)
        return self._pipes[mode]

    def unload(self) -> None:
        self._pipes.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- generation ------------------------------------------------------

    def resolve(self, req: GenerationRequest) -> dict[str, Any]:
        """Fill in defaults and snap sizes/frame counts to what the model accepts."""
        spec = self.spec
        mode = "i2v" if req.image is not None else "t2v"
        if mode not in spec.modes:
            raise ValueError(
                f"{spec.key} cannot start from an image; pick a model with image-to-video "
                f"support (for example wan2.2-5b or ltx-video)"
            )
        if req.last_image is not None and not spec.last_frame:
            raise ValueError(f"{spec.key} does not support a last-frame image")
        if req.last_image is not None and req.image is None:
            raise ValueError("a last-frame image needs a first-frame image too")
        fps = req.fps or spec.fps
        width, height = spec.snap_size(req.width or spec.width, req.height or spec.height)
        if req.num_frames:
            num_frames = spec.snap_frames(req.num_frames)
        elif req.duration:
            num_frames = spec.frames_for(req.duration, fps)
        else:
            num_frames = spec.num_frames
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        return {
            "mode": mode,
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": req.steps or spec.steps,
            "guidance": req.guidance if req.guidance is not None else spec.guidance,
            "seed": seed,
        }

    def generate(self, req: GenerationRequest, *, progress: ProgressFn | None = None) -> GenerationResult:
        import torch

        r = self.resolve(req)
        pipe = self.pipeline(r["mode"])

        kwargs: dict[str, Any] = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "height": r["height"],
            "width": r["width"],
            "num_frames": r["num_frames"],
            "num_inference_steps": r["steps"],
            "guidance_scale": r["guidance"],
            "frame_rate": r["fps"],
            "generator": torch.Generator(device="cpu").manual_seed(r["seed"]),
            "output_type": "np",
        }
        if req.image is not None:
            kwargs["image"] = fit_image(req.image, r["width"], r["height"])
        if req.last_image is not None:
            kwargs["last_image"] = fit_image(req.last_image, r["width"], r["height"])
        if progress is not None:
            total = r["steps"]

            def _on_step(_pipe: Any, step: int, _t: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                progress(step + 1, total)
                return callback_kwargs

            kwargs["callback_on_step_end"] = _on_step

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs, dropped = accepted_kwargs(pipe.__call__, kwargs)
        if dropped:
            log.debug("%s ignores: %s", type(pipe).__name__, ", ".join(dropped))

        output = pipe(**kwargs)
        frames = to_float_frames(output.frames[0] if hasattr(output, "frames") else output[0])

        audio, sample_rate = None, None
        raw_audio = getattr(output, "audio", None)
        if raw_audio is not None:
            audio = to_audio_array(raw_audio[0])
            vocoder = getattr(pipe, "vocoder", None)
            sample_rate = int(getattr(getattr(vocoder, "config", None), "output_sampling_rate", 24000))

        return GenerationResult(
            frames=frames,
            fps=r["fps"],
            seed=r["seed"],
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            model=self.spec.key,
            mode=r["mode"],
            width=frames.shape[2],
            height=frames.shape[1],
            steps=r["steps"],
            guidance=r["guidance"],
            audio=audio,
            sample_rate=sample_rate,
        )
