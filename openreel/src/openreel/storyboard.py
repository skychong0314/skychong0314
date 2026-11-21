"""Multi-shot storyboards: several shots rendered as one continuous video.

Seedance can generate a multi-shot sequence in one pass, keeping characters
and style consistent across cuts. Open models generate one shot at a time, so
openreel gets the same effect from four techniques:

1. **Cast sheet**: every named character is spelled out with the same
   description in every shot where they appear (``apply_cast``).
2. **Shared style and seed lineage**: one style preset for all shots and
   seeds derived from one base seed.
3. **Continuity chaining**: a shot marked ``continue`` starts from the
   previous shot's last frame through image-to-video, so motion carries on
   with no visible cut.
4. **Reference frames**: a shot can start from an image you supply, which is
   the strongest way to pin down a character's look.

Shots are concatenated with hard cuts, except continued shots, where the
duplicated first frame is dropped so the motion does not stutter.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .generator import GenerationRequest, GenerationResult, VideoGenerator
from .media import fit_audio, frame_to_image, load_image, save_video, to_uint8
from .models import ModelSpec
from .prompting import DEFAULT_NEGATIVE_PROMPT, apply_cast, compose_prompt

EnhanceFn = Callable[[str, bool], str]
ProgressFn = Callable[[float, str], None]

SETTING_KEYS = {"title", "model", "style", "negative", "seed", "fps", "steps", "width", "height", "guidance"}
_INT_KEYS = {"seed", "fps", "steps", "width", "height"}


@dataclass
class Shot:
    prompt: str
    duration: float | None = None
    num_frames: int | None = None
    camera: str | None = None
    transition: str = "cut"
    image: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.transition not in ("cut", "continue"):
            raise ValueError(f"transition must be 'cut' or 'continue', got {self.transition!r}")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("every shot needs a prompt")

    @property
    def continues(self) -> bool:
        return self.transition == "continue"


@dataclass
class Storyboard:
    shots: list[Shot]
    title: str | None = None
    model: str | None = None
    style: str | None = None
    cast: dict[str, str] = field(default_factory=dict)
    negative: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    steps: int | None = None
    guidance: float | None = None
    base_dir: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> Storyboard:
        data = dict(data)
        raw_shots = data.pop("shots", None)
        if not raw_shots:
            raise ValueError("a storyboard needs at least one shot")
        shots = []
        for i, s in enumerate(raw_shots, 1):
            try:
                shots.append(Shot(prompt=s) if isinstance(s, str) else Shot(**s))
            except TypeError as exc:
                raise ValueError(f"shot {i}: {exc}") from None
        unknown = set(data) - (SETTING_KEYS | {"cast", "negative_prompt"})
        if unknown:
            raise ValueError(f"unknown storyboard keys: {', '.join(sorted(unknown))}")
        if "negative_prompt" in data:
            data["negative"] = data.pop("negative_prompt")
        return cls(shots=shots, base_dir=base_dir or Path.cwd(), **data)

    @classmethod
    def parse_script(cls, text: str, base_dir: Path | None = None) -> Storyboard:
        """Parse the compact one-line-per-shot script format.

        ::

            title: Rainy night
            style: cinematic
            cast: Mira = a young woman with short silver hair and a red raincoat
            [4s] Mira walks through a neon-lit alley at night | tracking
            [+3s] Mira stops and looks up into the rain
            [3s] close-up of Mira smiling | dolly-in

        ``[4s]`` sets a shot's length, a leading ``+`` continues from the
        previous shot's last frame, and ``| name`` picks a camera move.
        """
        settings: dict[str, Any] = {}
        cast: dict[str, str] = {}
        shots: list[Shot] = []
        shot_re = re.compile(r"^\[(?P<plus>\+)?\s*(?:(?P<secs>\d+(?:\.\d+)?)\s*s)?\s*\]\s*(?P<rest>.*)$")
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition(":")
            key = key.strip().lower()
            if sep and key == "cast":
                name, eq, desc = value.partition("=")
                if not eq or not name.strip() or not desc.strip():
                    raise ValueError(f"line {lineno}: cast lines look like 'cast: Name = description'")
                cast[name.strip()] = desc.strip()
                continue
            if sep and key in SETTING_KEYS:
                value = value.strip()
                if key in _INT_KEYS:
                    settings[key] = int(value)
                elif key == "guidance":
                    settings[key] = float(value)
                else:
                    settings[key] = value
                continue
            transition, duration, rest = "cut", None, line
            m = shot_re.match(line)
            if m:
                transition = "continue" if m.group("plus") else "cut"
                duration = float(m.group("secs")) if m.group("secs") else None
                rest = m.group("rest")
            prompt, _, camera = rest.partition("|")
            shots.append(
                Shot(
                    prompt=prompt.strip(),
                    duration=duration,
                    camera=camera.strip() or None,
                    transition=transition,
                )
            )
        if not shots:
            raise ValueError("the script has no shot lines")
        return cls(shots=shots, cast=cast, base_dir=base_dir or Path.cwd(), **settings)

    @classmethod
    def load(cls, path: str | Path) -> Storyboard:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            return cls.from_dict(yaml.safe_load(text), base_dir=path.parent)
        if path.suffix.lower() == ".json":
            return cls.from_dict(json.loads(text), base_dir=path.parent)
        return cls.parse_script(text, base_dir=path.parent)


@dataclass
class PlannedShot:
    index: int
    shot: Shot
    prompt: str
    seed: int
    num_frames: int


def plan_storyboard(
    board: Storyboard,
    spec: ModelSpec,
    *,
    enhance: EnhanceFn | None = None,
) -> list[PlannedShot]:
    """Validate a storyboard against a model and build each shot's final prompt.

    Runs before any video model loads, so mistakes surface immediately and an
    LLM rewriter can be unloaded before generation starts.
    """
    needs_i2v = [i + 1 for i, s in enumerate(board.shots) if s.continues or s.image]
    if needs_i2v and "i2v" not in spec.modes:
        raise ValueError(
            f"shots {needs_i2v} continue from a frame or use a reference image, which needs "
            f"image-to-video, and {spec.key} is text-to-video only"
        )
    if board.shots[0].continues:
        raise ValueError("the first shot has nothing to continue from")

    base_seed = board.seed if board.seed is not None else random.randrange(2**31 - 1000)
    fps = board.fps or spec.fps
    planned = []
    for i, shot in enumerate(board.shots):
        text = shot.prompt
        if enhance is not None:
            text = enhance(text, bool(shot.continues or shot.image))
        prompt = compose_prompt(apply_cast(text, board.cast), style=board.style, camera=shot.camera)
        if shot.num_frames:
            num_frames = spec.snap_frames(shot.num_frames)
        elif shot.duration:
            num_frames = spec.frames_for(shot.duration, fps)
        else:
            num_frames = spec.num_frames
        seed = shot.seed if shot.seed is not None else base_seed + i
        planned.append(PlannedShot(index=i, shot=shot, prompt=prompt, seed=seed, num_frames=num_frames))
    return planned


@dataclass
class StoryboardResult:
    frames: np.ndarray
    fps: int
    shots: list[GenerationResult]
    plans: list[PlannedShot]
    title: str | None = None
    audio: np.ndarray | None = None
    sample_rate: int | None = None

    @property
    def duration(self) -> float:
        return self.frames.shape[0] / self.fps

    def manifest(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "model": self.shots[0].model,
            "fps": self.fps,
            "num_frames": int(self.frames.shape[0]),
            "duration_s": round(self.duration, 3),
            "shots": [
                {
                    "index": p.index + 1,
                    "transition": p.shot.transition,
                    "source_prompt": p.shot.prompt,
                    **r.metadata(),
                }
                for p, r in zip(self.plans, self.shots)
            ],
        }

    def save(self, path: str | Path, *, keep_shots: bool = False) -> Path:
        path = save_video(path, self.frames, self.fps, audio=self.audio, sample_rate=self.sample_rate)
        path.with_suffix(".json").write_text(json.dumps(self.manifest(), indent=2) + "\n")
        if keep_shots:
            for i, shot in enumerate(self.shots, 1):
                shot.save(path.with_name(f"{path.stem}_shot{i:02d}{path.suffix}"))
        return path


def _resize_frames(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    if frames.shape[1:3] == (height, width):
        return frames
    resized = [
        np.asarray(Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS))
        for f in to_uint8(frames)
    ]
    return np.stack(resized).astype(np.float32) / 255.0


def render_storyboard(
    board: Storyboard,
    plans: list[PlannedShot],
    generator: VideoGenerator,
    *,
    progress: ProgressFn | None = None,
) -> StoryboardResult:
    spec = generator.spec
    fps = board.fps or spec.fps
    negative = board.negative or DEFAULT_NEGATIVE_PROMPT
    n = len(plans)
    results: list[GenerationResult] = []

    for plan in plans:
        shot = plan.shot
        image = None
        if shot.continues:
            image = frame_to_image(results[-1].frames, -1)
        elif shot.image:
            image = load_image(board.base_dir / shot.image)
        if progress:
            progress(plan.index / n, f"Shot {plan.index + 1}/{n}: {shot.prompt[:60]}")

        def step_progress(step: int, total: int, i: int = plan.index) -> None:
            if progress:
                progress((i + step / total) / n, f"Shot {i + 1}/{n}: step {step}/{total}")

        req = GenerationRequest(
            prompt=plan.prompt,
            negative_prompt=negative,
            image=image,
            width=board.width,
            height=board.height,
            num_frames=plan.num_frames,
            fps=fps,
            steps=board.steps,
            guidance=board.guidance,
            seed=plan.seed,
        )
        results.append(generator.generate(req, progress=step_progress))

    height, width = results[0].frames.shape[1:3]
    pieces = []
    for plan, res in zip(plans, results):
        frames = _resize_frames(res.frames, width, height)
        pieces.append(frames[1:] if plan.shot.continues else frames)
    frames = np.concatenate(pieces, axis=0)

    audio, sample_rate = None, None
    if all(r.audio is not None for r in results):
        sample_rate = results[0].sample_rate
        per_frame = sample_rate / fps
        tracks = []
        for plan, res in zip(plans, results):
            track = fit_audio(res.audio, round(res.num_frames * per_frame))
            if plan.shot.continues:
                track = track[:, round(per_frame) :]
            tracks.append(track)
        channels = max(t.shape[0] for t in tracks)
        tracks = [np.repeat(t, channels, axis=0) if t.shape[0] == 1 and channels > 1 else t for t in tracks]
        audio = np.concatenate(tracks, axis=1)

    if progress:
        progress(1.0, "Done")
    return StoryboardResult(
        frames=frames,
        fps=fps,
        shots=results,
        plans=plans,
        title=board.title,
        audio=audio,
        sample_rate=sample_rate,
    )
