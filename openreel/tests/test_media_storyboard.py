import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openreel.generator import GenerationResult, VideoGenerator
from openreel.media import fit_audio, fit_image, save_video, to_float_frames, to_uint8
from openreel.prompting import CAMERA_MOVES, STYLES
from openreel.storyboard import Shot, Storyboard, plan_storyboard, render_storyboard

from .conftest import tiny_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _probe(path: Path) -> str:
    import imageio_ffmpeg

    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True)
    return proc.stderr


# -- media -----------------------------------------------------------------

def test_to_float_frames_normalises_layouts():
    u8 = np.full((2, 4, 6, 3), 255, dtype=np.uint8)
    assert to_float_frames(u8).max() == 1.0
    chw = np.zeros((2, 3, 4, 6), dtype=np.float32)
    assert to_float_frames(chw).shape == (2, 4, 6, 3)
    batched = np.zeros((1, 2, 4, 6, 3), dtype=np.float32)
    assert to_float_frames(batched).shape == (2, 4, 6, 3)
    pil = [Image.new("RGB", (6, 4), (255, 0, 0))] * 2
    out = to_float_frames(pil)
    assert out.shape == (2, 4, 6, 3) and out[0, 0, 0, 0] == 1.0


def test_fit_image_exact_size(image):
    assert fit_image(image, 32, 16).size == (32, 16)
    assert fit_image(image.convert("L"), 16, 16).mode == "RGB"


def test_fit_audio():
    a = np.ones((2, 10), dtype=np.float32)
    assert fit_audio(a, 4).shape == (2, 4)
    padded = fit_audio(a, 15)
    assert padded.shape == (2, 15) and padded[:, 10:].sum() == 0


def test_save_video_with_audio(tmp_path):
    frames = np.random.default_rng(0).random((24, 32, 48, 3)).astype(np.float32)
    t = np.linspace(0, 1, 16000, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32) * 0.3
    out = save_video(tmp_path / "av.mp4", frames, 24, audio=audio, sample_rate=16000)
    info = _probe(out)
    assert "Video: h264" in info and "Audio: aac" in info
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    assert float(s) == pytest.approx(1.0, abs=0.1)
    with pytest.raises(ValueError, match="sample_rate"):
        save_video(tmp_path / "x.mp4", frames, 24, audio=audio)


# -- storyboard parsing and planning ---------------------------------------

def test_parse_script():
    board = Storyboard.parse_script(
        """
        # comment
        title: Test
        style: anime
        seed: 5
        cast: Mira = silver hair
        [4s] Mira walks | tracking
        [+2.5s] Mira stops
        close-up: Mira smiles
        [+] she waves | orbit
        """
    )
    assert (board.title, board.style, board.seed, board.cast) == ("Test", "anime", 5, {"Mira": "silver hair"})
    assert [s.duration for s in board.shots] == [4.0, 2.5, None, None]
    assert [s.transition for s in board.shots] == ["cut", "continue", "cut", "continue"]
    assert [s.camera for s in board.shots] == ["tracking", None, None, "orbit"]
    assert board.shots[2].prompt == "close-up: Mira smiles"


def test_script_errors():
    with pytest.raises(ValueError, match="no shot lines"):
        Storyboard.parse_script("title: x\n")
    with pytest.raises(ValueError, match="cast lines"):
        Storyboard.parse_script("cast: Mira\n[2s] hi")
    with pytest.raises(ValueError, match="unknown storyboard keys"):
        Storyboard.from_dict({"shots": ["a"], "colour": "red"})
    with pytest.raises(ValueError, match="shot 1"):
        Storyboard.from_dict({"shots": [{"prompt": "a", "lens": "50mm"}]})


def test_yaml_and_script_examples_match():
    y = Storyboard.load(EXAMPLES / "rainy_night.yaml")
    s = Storyboard.load(EXAMPLES / "rainy_night.txt")
    for field in ("title", "model", "style", "seed", "cast"):
        assert getattr(y, field) == getattr(s, field), field
    assert [(a.prompt, a.duration, a.camera, a.transition) for a in y.shots] == \
           [(b.prompt, b.duration, b.camera, b.transition) for b in s.shots]


def test_plan_builds_consistent_prompts():
    board = Storyboard.load(EXAMPLES / "rainy_night.yaml")
    spec = tiny_spec(fps=24)
    plans = plan_storyboard(board, spec)
    desc = "Mira (a young woman with short silver hair and a red raincoat)"
    assert all(desc in p.prompt and STYLES["cinematic"] in p.prompt for p in plans)
    assert CAMERA_MOVES["tracking"] in plans[0].prompt
    assert [p.seed for p in plans] == [42, 43, 44]
    assert [p.num_frames for p in plans] == [97, 73, 73]


def test_plan_with_enhancer_and_validation():
    board = Storyboard(shots=[Shot("a fox"), Shot("it runs", transition="continue")])
    calls = []
    plans = plan_storyboard(board, tiny_spec(), enhance=lambda t, i2v: calls.append((t, i2v)) or t.upper())
    assert calls == [("a fox", False), ("it runs", True)]
    assert plans[0].prompt.startswith("A FOX.")
    with pytest.raises(ValueError, match="text-to-video only"):
        plan_storyboard(board, tiny_spec(i2v_repo=None, i2v_pipeline=None))
    with pytest.raises(ValueError, match="nothing to continue"):
        plan_storyboard(Storyboard(shots=[Shot("x", transition="continue")]), tiny_spec())
    with pytest.raises(ValueError, match="camera move"):
        plan_storyboard(Storyboard(shots=[Shot("x", camera="whip")]), tiny_spec())
    with pytest.raises(ValueError, match="transition"):
        Shot("x", transition="fade")


# -- rendering ---------------------------------------------------------------

def test_render_chains_last_frame_into_continued_shot(generator, tmp_path):
    board = Storyboard(
        shots=[Shot("a fox in snow"), Shot("the fox runs", transition="continue"), Shot("a dog at night")],
        seed=10,
        cast={"fox": "orange fur"},
    )
    plans = plan_storyboard(board, generator.spec)
    requests = []
    original = generator.generate

    def spy(req, progress=None):
        requests.append(req)
        return original(req, progress=progress)

    generator.generate = spy
    messages = []
    result = render_storyboard(board, plans, generator, progress=lambda f, m: messages.append((f, m)))

    assert requests[0].image is None and requests[2].image is None
    np.testing.assert_array_equal(np.asarray(requests[1].image), to_uint8(result.shots[0].frames[-1]))
    assert result.frames.shape == (9 + 8 + 9, 32, 32, 3)
    np.testing.assert_array_equal(result.frames[9:17], result.shots[1].frames[1:])
    assert [r.seed for r in result.shots] == [10, 11, 12]
    assert messages[-1] == (1.0, "Done")
    assert all(0.0 <= f <= 1.0 for f, _ in messages)

    out = result.save(tmp_path / "story.mp4", keep_shots=True)
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["num_frames"] == 26 and len(manifest["shots"]) == 3
    assert manifest["shots"][1]["transition"] == "continue"
    assert manifest["shots"][1]["mode"] == "i2v"
    assert sorted(p.name for p in tmp_path.glob("story_shot*.mp4")) == \
           ["story_shot01.mp4", "story_shot02.mp4", "story_shot03.mp4"]


def test_render_concatenates_audio_in_sync():
    class FakeGen:
        spec = tiny_spec(fps=8)

        def generate(self, req, progress=None):
            n = req.num_frames
            return GenerationResult(
                frames=np.zeros((n, 16, 16, 3), np.float32), fps=8, seed=req.seed, prompt=req.prompt,
                negative_prompt=None, model="fake", mode="i2v" if req.image else "t2v", width=16, height=16,
                steps=1, guidance=None, audio=np.ones((1, 1000), np.float32), sample_rate=800,
            )

    board = Storyboard(shots=[Shot("a"), Shot("b", transition="continue")], seed=1)
    plans = plan_storyboard(board, FakeGen.spec)
    result = render_storyboard(board, plans, FakeGen())
    assert result.frames.shape[0] == 17
    assert result.audio.shape == (1, round(17 * 800 / 8))


def test_resolution_mismatch_is_resized(spec, t2v_pipe, i2v_pipe):
    gen = VideoGenerator(spec, t2v_pipe=t2v_pipe, i2v_pipe=i2v_pipe)
    board = Storyboard(shots=[Shot("a"), Shot("b")], seed=1)
    plans = plan_storyboard(board, spec)
    original = gen.generate

    def second_is_small(req, progress=None):
        if req.seed == 2:
            req.width, req.height = 16, 16
        return original(req, progress=progress)

    gen.generate = second_is_small
    result = render_storyboard(board, plans, gen)
    assert result.shots[1].frames.shape[1:3] == (16, 16)
    assert result.frames.shape == (18, 32, 32, 3)
