"""Command-line interface: ``openreel generate | story | models | presets | inspect | app``."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .models import DEFAULT_MODEL, MODELS, ModelSpec, get_model
from .prompting import CAMERA_MOVES, DEFAULT_LLM, DEFAULT_NEGATIVE_PROMPT, STYLES, compose_prompt

GeneratorFactory = Callable[[ModelSpec, argparse.Namespace], Any]


def _default_output(prompt: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "video"
    return Path("outputs") / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.mp4"


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    p.add_argument("--offload", choices=["model", "sequential", "none"], default="model",
                   help="CUDA memory strategy: 'model' (default) moves whole models on and off the GPU, "
                        "'sequential' streams layers (lowest VRAM, slowest), 'none' keeps all on GPU")
    p.add_argument("--no-tiling", action="store_true", help="disable tiled VAE decoding")
    p.add_argument("--enhance", choices=["none", "llm"], default="none",
                   help="'llm' rewrites prompts with a local open LLM before generating")
    p.add_argument("--llm", default=DEFAULT_LLM, help=f"LLM for --enhance llm (default {DEFAULT_LLM})")
    p.add_argument("--dry-run", action="store_true", help="print the final prompts and settings, generate nothing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openreel", description="Seedance-style video generation with open models.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="text-to-video, or image-to-video with --image")
    g.add_argument("prompt")
    g.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    g.add_argument("-i", "--image", type=Path, help="first frame (switches to image-to-video)")
    g.add_argument("--last-image", type=Path, help="last frame (models with first+last-frame support)")
    g.add_argument("-s", "--style", choices=list(STYLES))
    g.add_argument("-c", "--camera", choices=list(CAMERA_MOVES))
    g.add_argument("--raw", action="store_true", help="send the prompt exactly as written")
    g.add_argument("-n", "--negative", default=DEFAULT_NEGATIVE_PROMPT)
    length = g.add_mutually_exclusive_group()
    length.add_argument("-d", "--duration", type=float, help="seconds")
    length.add_argument("--frames", type=int)
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--fps", type=int)
    g.add_argument("--steps", type=int)
    g.add_argument("--guidance", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("-o", "--output", type=Path)
    _add_runtime_args(g)

    s = sub.add_parser("story", help="render a multi-shot storyboard (.yaml, .json or script .txt)")
    s.add_argument("file", type=Path)
    s.add_argument("-m", "--model", choices=list(MODELS), help="overrides the storyboard's model")
    s.add_argument("--seed", type=int, help="overrides the storyboard's base seed")
    s.add_argument("--keep-shots", action="store_true", help="also save each shot as its own file")
    s.add_argument("-o", "--output", type=Path)
    _add_runtime_args(s)

    sub.add_parser("models", help="list supported models")
    sub.add_parser("presets", help="list style presets and camera moves")

    i = sub.add_parser("inspect", help="show a model's architecture from its safetensors headers")
    i.add_argument("source", help="openreel model key, Hugging Face repo id, local folder or .safetensors file")
    i.add_argument("--json", action="store_true")

    a = sub.add_parser("app", help="launch the web UI")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=7860)
    a.add_argument("--share", action="store_true", help="create a public Gradio link")
    a.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    _add_runtime_args(a)
    return parser


def default_generator_factory(spec: ModelSpec, args: argparse.Namespace) -> Any:
    from .generator import VideoGenerator

    return VideoGenerator(spec, device=args.device, offload=args.offload, vae_tiling=not args.no_tiling)


def _make_enhancer(args: argparse.Namespace) -> tuple[Callable[[str, bool], str] | None, Any]:
    if args.enhance != "llm":
        return None, None
    from .prompting import LLMRewriter

    rewriter = LLMRewriter(args.llm)
    return (lambda text, i2v: rewriter.rewrite(text, image_to_video=i2v)), rewriter


def cmd_models() -> int:
    for spec in MODELS.values():
        modes = " + ".join(m.upper() for m in spec.modes)
        extras = [x for x, on in (("audio", spec.audio), ("first+last frame", spec.last_frame)) if on]
        print(f"{spec.key:<12} {spec.name}")
        print(f"{'':<12} {modes}{' · ' + ', '.join(extras) if extras else ''} · "
              f"{spec.width}x{spec.height} @ {spec.fps} fps · ~{spec.vram_gb} GB VRAM")
        print(f"{'':<12} {spec.summary}")
        if spec.notes:
            print(f"{'':<12} note: {spec.notes}")
    print(f"\ndefault: {DEFAULT_MODEL}")
    return 0


def cmd_presets() -> int:
    print("styles (--style):")
    for k, v in STYLES.items():
        print(f"  {k:<13} {v}")
    print("\ncamera moves (--camera):")
    for k, v in CAMERA_MOVES.items():
        print(f"  {k:<13} {v}")
    return 0


def cmd_generate(args: argparse.Namespace, factory: GeneratorFactory) -> int:
    from .generator import GenerationRequest
    from .media import load_image

    spec = get_model(args.model)
    is_i2v = args.image is not None
    text = args.prompt
    enhance, rewriter = _make_enhancer(args)
    if enhance is not None:
        print("rewriting prompt with", args.llm, file=sys.stderr)
        text = enhance(text, is_i2v)
        rewriter.unload()
    prompt = text if args.raw else compose_prompt(text, style=args.style, camera=args.camera)

    req = GenerationRequest(
        prompt=prompt,
        negative_prompt=args.negative or None,
        image=load_image(args.image) if is_i2v else None,
        last_image=load_image(args.last_image) if args.last_image else None,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        duration=args.duration,
        fps=args.fps,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )
    gen = factory(spec, args)
    if args.dry_run:
        print(json.dumps({"model": spec.key, "prompt": prompt, **gen.resolve(req)}, indent=2))
        return 0
    result = gen.generate(req)
    out = result.save(args.output or _default_output(args.prompt))
    print(f"saved {out}  ({result.width}x{result.height}, {result.num_frames} frames, "
          f"{result.duration:.1f}s, seed {result.seed})")
    return 0


def cmd_story(args: argparse.Namespace, factory: GeneratorFactory) -> int:
    from .storyboard import Storyboard, plan_storyboard, render_storyboard

    board = Storyboard.load(args.file)
    if args.seed is not None:
        board.seed = args.seed
    spec = get_model(args.model or board.model or DEFAULT_MODEL)
    enhance, rewriter = _make_enhancer(args)
    plans = plan_storyboard(board, spec, enhance=enhance)
    if rewriter is not None:
        rewriter.unload()

    fps = board.fps or spec.fps
    total = sum(p.num_frames for p in plans) - sum(1 for p in plans if p.shot.continues)
    print(f"{board.title or args.file.stem}: {len(plans)} shots, ~{total / fps:.1f}s on {spec.key}")
    for p in plans:
        tag = "continue" if p.shot.continues else ("image" if p.shot.image else "cut")
        print(f"  shot {p.index + 1} [{tag}, {p.num_frames} frames, seed {p.seed}] {p.prompt}")
    if args.dry_run:
        return 0

    gen = factory(spec, args)
    last_msg = [""]

    def progress(_fraction: float, message: str) -> None:
        if "step" not in message and message != last_msg[0]:
            print(message, file=sys.stderr)
            last_msg[0] = message

    result = render_storyboard(board, plans, gen, progress=progress)
    out = result.save(args.output or _default_output(board.title or args.file.stem), keep_shots=args.keep_shots)
    print(f"saved {out}  ({result.frames.shape[0]} frames, {result.duration:.1f}s)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .inspect_model import format_report, inspect_model

    try:
        report = inspect_model(args.source)
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:  # network or Hub errors: report them without a traceback
        print(f"error: could not read {args.source}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2) if args.json else format_report(report))
    return 0


def cmd_app(args: argparse.Namespace, factory: GeneratorFactory) -> int:
    from .app import OUTPUT_DIR, build_app

    demo = build_app(default_model=args.model, factory=lambda spec: factory(spec, args), runtime_args=args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        allowed_paths=[str(OUTPUT_DIR.resolve())])
    return 0


def main(argv: Sequence[str] | None = None, *, factory: GeneratorFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("openreel").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    factory = factory or default_generator_factory
    try:
        if args.command == "models":
            return cmd_models()
        if args.command == "presets":
            return cmd_presets()
        if args.command == "generate":
            return cmd_generate(args, factory)
        if args.command == "story":
            return cmd_story(args, factory)
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "app":
            return cmd_app(args, factory)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
