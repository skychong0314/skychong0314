"""Gradio web UI with a single-shot tab and a multi-shot storyboard tab."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .models import DEFAULT_MODEL, MODELS, ModelSpec, get_model
from .prompting import CAMERA_MOVES, DEFAULT_LLM, DEFAULT_NEGATIVE_PROMPT, STYLES, LLMRewriter, compose_prompt

OUTPUT_DIR = Path("outputs")

EXAMPLE_SCRIPT = """\
title: Rainy night
style: cinematic
cast: Mira = a young woman with short silver hair and a red raincoat
[4s] Mira walks through a neon-lit alley at night, rain falling | tracking
[+3s] Mira stops and looks up into the rain
[3s] close-up of Mira smiling, raindrops on her face | dolly-in
"""

NONE = "(none)"


class GeneratorCache:
    """Keeps one model loaded at a time; switching models frees the previous one."""

    def __init__(self, factory: Callable[[ModelSpec], Any]) -> None:
        self.factory = factory
        self.key: str | None = None
        self.gen: Any = None
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        if key != self.key:
            if self.gen is not None:
                self.gen.unload()
            self.gen = self.factory(get_model(key))
            self.key = key
        return self.gen


def _out_path(stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}.mp4"


def _opt(value: str | None) -> str | None:
    return None if value in (None, "", NONE) else value


def run_generate(
    cache: GeneratorCache,
    prompt: str,
    image: Any = None,
    last_image: Any = None,
    model_key: str = DEFAULT_MODEL,
    style: str | None = None,
    camera: str | None = None,
    raw: bool = False,
    use_llm: bool = False,
    negative: str = DEFAULT_NEGATIVE_PROMPT,
    duration: float | None = None,
    seed: int = -1,
    steps: int | None = None,
    llm_id: str = DEFAULT_LLM,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    from .generator import GenerationRequest

    if not prompt or not prompt.strip():
        raise ValueError("Write a prompt first.")
    text = prompt
    if use_llm:
        if progress:
            progress(0.0, "Rewriting prompt with the LLM")
        rewriter = LLMRewriter(llm_id)
        text = rewriter.rewrite(prompt, image_to_video=image is not None)
        rewriter.unload()
    final = text if raw else compose_prompt(text, style=_opt(style), camera=_opt(camera))
    req = GenerationRequest(
        prompt=final,
        negative_prompt=negative or None,
        image=image,
        last_image=last_image,
        duration=duration or None,
        steps=int(steps) if steps else None,
        seed=None if seed is None or int(seed) < 0 else int(seed),
    )
    with cache.lock:
        gen = cache.get(model_key)
        step_cb = (lambda s, t: progress(s / t, f"Denoising step {s}/{t}")) if progress else None
        result = gen.generate(req, progress=step_cb)
    path = result.save(_out_path(f"{model_key}_{result.seed}"))
    return str(path), result.metadata()


def run_story(
    cache: GeneratorCache,
    script: str,
    model_key: str = DEFAULT_MODEL,
    seed: int = -1,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    from .storyboard import Storyboard, plan_storyboard, render_storyboard

    board = Storyboard.parse_script(script, base_dir=Path.cwd())
    if seed is not None and int(seed) >= 0:
        board.seed = int(seed)
    spec = get_model(model_key)
    plans = plan_storyboard(board, spec)
    with cache.lock:
        gen = cache.get(model_key)
        result = render_storyboard(board, plans, gen, progress=progress)
    path = result.save(_out_path(f"story_{model_key}"))
    return str(path), result.manifest()


def build_app(
    default_model: str = DEFAULT_MODEL,
    factory: Callable[[ModelSpec], Any] | None = None,
    runtime_args: Any = None,
) -> Any:
    import gradio as gr

    if factory is None:
        from .generator import VideoGenerator

        factory = VideoGenerator
    cache = GeneratorCache(factory)
    llm_id = getattr(runtime_args, "llm", DEFAULT_LLM)
    model_choices = [(f"{s.name} — {s.key}", s.key) for s in MODELS.values()]
    i2v_note = ", ".join(k for k, s in MODELS.items() if "i2v" in s.modes)

    def gr_progress(p: Any) -> Callable[[float, str], None]:
        return lambda fraction, message: p(fraction, desc=message)

    def on_generate(prompt, image, last_image, model_key, style, camera, raw, use_llm, negative,
                    duration, seed, steps, progress=gr.Progress()):
        try:
            path, meta = run_generate(cache, prompt, image, last_image, model_key, style, camera, raw,
                                      use_llm, negative, duration, seed, steps, llm_id, gr_progress(progress))
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        return path, json.dumps(meta, indent=2)

    def on_story(script, model_key, seed, progress=gr.Progress()):
        try:
            path, manifest = run_story(cache, script, model_key, seed, gr_progress(progress))
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        return path, json.dumps(manifest, indent=2)

    with gr.Blocks(title="openreel") as demo:
        gr.Markdown(
            "# openreel\nSeedance-style video generation with open models. "
            f"Image-to-video works with: {i2v_note}."
        )
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", lines=4,
                                        placeholder="A red fox trots through fresh snow at dawn")
                    with gr.Row():
                        image = gr.Image(label="First frame (optional)", type="pil")
                        last_image = gr.Image(label="Last frame (optional)", type="pil")
                    model = gr.Dropdown(model_choices, value=default_model, label="Model")
                    with gr.Row():
                        style = gr.Dropdown([NONE, *STYLES], value="cinematic", label="Style")
                        camera = gr.Dropdown([NONE, *CAMERA_MOVES], value=NONE, label="Camera")
                    with gr.Accordion("Advanced", open=False):
                        use_llm = gr.Checkbox(label=f"Rewrite prompt with LLM ({llm_id})", value=False)
                        raw = gr.Checkbox(label="Send prompt exactly as written", value=False)
                        negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)
                        duration = gr.Slider(0, 10, value=0, step=0.5,
                                             label="Duration in seconds (0 = model default)")
                        seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        steps = gr.Slider(0, 80, value=0, step=1, label="Steps (0 = model default)")
                    go = gr.Button("Generate", variant="primary")
                with gr.Column():
                    video = gr.Video(label="Result")
                    info = gr.Code(label="Settings used", language="json")
            go.click(on_generate, [prompt, image, last_image, model, style, camera, raw, use_llm, negative,
                                   duration, seed, steps], [video, info])

        with gr.Tab("Storyboard"):
            with gr.Row():
                with gr.Column():
                    script = gr.Code(value=EXAMPLE_SCRIPT, label="Script", lines=14)
                    gr.Markdown(
                        "`[4s]` sets a shot's length · `[+3s]` continues from the previous shot's last frame "
                        "(needs an image-to-video model) · `| dolly-in` adds a camera move · "
                        "`cast: Name = description` keeps a character consistent across shots."
                    )
                    story_model = gr.Dropdown(model_choices, value=default_model, label="Model")
                    story_seed = gr.Number(label="Base seed (-1 = random)", value=-1, precision=0)
                    story_go = gr.Button("Render storyboard", variant="primary")
                with gr.Column():
                    story_video = gr.Video(label="Result")
                    story_info = gr.Code(label="Shot manifest", language="json")
            story_go.click(on_story, [script, story_model, story_seed], [story_video, story_info])

    return demo
