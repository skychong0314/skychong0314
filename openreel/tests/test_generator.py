import inspect
import json
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from diffusers import WanImageToVideoPipeline

from openreel.generator import GenerationRequest, VideoGenerator, accepted_kwargs

from .conftest import tiny_spec


def test_text_to_video(generator):
    res = generator.generate(GenerationRequest(prompt="a cat walks on the beach", seed=7))
    assert res.frames.shape == (9, 32, 32, 3)
    assert res.frames.dtype == np.float32
    assert 0.0 <= res.frames.min() and res.frames.max() <= 1.0
    assert (res.mode, res.seed, res.fps, res.steps) == ("t2v", 7, 8, 2)
    assert res.duration == pytest.approx(9 / 8)


def test_seed_is_reproducible(generator):
    a = generator.generate(GenerationRequest(prompt="a dog runs", seed=123))
    b = generator.generate(GenerationRequest(prompt="a dog runs", seed=123))
    c = generator.generate(GenerationRequest(prompt="a dog runs", seed=124))
    np.testing.assert_array_equal(a.frames, b.frames)
    assert not np.array_equal(a.frames, c.frames)


def test_duration_and_size_are_snapped(generator):
    r = generator.resolve(GenerationRequest(prompt="x", duration=2, width=40, height=36))
    assert (r["num_frames"], r["width"], r["height"]) == (17, 32, 32)
    res = generator.generate(GenerationRequest(prompt="x", num_frames=12, seed=1))
    assert res.num_frames == 13


def test_image_to_video_and_last_frame(generator, image):
    res = generator.generate(GenerationRequest(prompt="a fox", image=image, seed=3))
    assert res.mode == "i2v" and res.frames.shape == (9, 32, 32, 3)
    res = generator.generate(GenerationRequest(prompt="a fox", image=image, last_image=image, seed=3))
    assert res.frames.shape == (9, 32, 32, 3)


def test_mode_errors(t2v_pipe, image):
    t2v_only = VideoGenerator(tiny_spec(i2v_repo=None, i2v_pipeline=None, last_frame=False), t2v_pipe=t2v_pipe)
    with pytest.raises(ValueError, match="cannot start from an image"):
        t2v_only.generate(GenerationRequest(prompt="x", image=image))
    no_last = VideoGenerator(tiny_spec(last_frame=False), t2v_pipe=t2v_pipe)
    with pytest.raises(ValueError, match="last-frame"):
        no_last.resolve(GenerationRequest(prompt="x", image=image, last_image=image))
    with pytest.raises(ValueError, match="first-frame image"):
        VideoGenerator(tiny_spec()).resolve(GenerationRequest(prompt="x", last_image=image))


def test_progress_callback(generator):
    seen = []
    generator.generate(GenerationRequest(prompt="x", seed=1), progress=lambda s, t: seen.append((s, t)))
    assert seen == [(1, 2), (2, 2)]


def test_ti2v_reuses_weights_via_from_pipe(ti2v_t2v_pipe, image):
    spec = tiny_spec(i2v_repo="local/tiny", last_frame=False)  # same repo for T2V and I2V
    gen = VideoGenerator(spec, device="cpu", t2v_pipe=ti2v_t2v_pipe)
    res = gen.generate(GenerationRequest(prompt="a fox", image=image, seed=5))
    i2v = gen._pipes["i2v"]
    assert isinstance(i2v, WanImageToVideoPipeline)
    assert i2v.config.expand_timesteps is True
    assert i2v.transformer is ti2v_t2v_pipe.transformer
    assert res.frames.shape == (9, 32, 32, 3)


class RecordingPipe:
    """Mimics HunyuanVideo 1.5 I2V: no height/width/guidance_scale/frame_rate parameters."""

    def __init__(self, audio=False):
        self.calls = []
        if audio:
            self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=16000))
        self.audio = audio

    def __call__(self, image, prompt=None, negative_prompt=None, num_frames=121, num_inference_steps=50,
                 generator=None, output_type="np", callback_on_step_end=None):
        self.calls.append(dict(image=image, prompt=prompt, num_frames=num_frames))
        frames = np.full((1, num_frames, 16, 24, 3), 0.5, dtype=np.float32)
        audio = torch.zeros(1, 2, 16000) if self.audio else None
        return SimpleNamespace(frames=frames, audio=audio)


def test_kwargs_are_filtered_per_pipeline(image):
    pipe = RecordingPipe(audio=True)
    gen = VideoGenerator(tiny_spec(), i2v_pipe=pipe)
    res = gen.generate(GenerationRequest(prompt="hi", image=image, seed=1), progress=lambda s, t: None)
    assert pipe.calls[0]["num_frames"] == 9
    assert res.frames.shape == (9, 16, 24, 3)
    assert (res.width, res.height) == (24, 16)
    assert res.audio.shape == (2, 16000) and res.sample_rate == 16000


def test_accepted_kwargs_respects_var_keyword():
    def f(a, **kw):
        pass

    def g(a, b=1):
        pass

    assert accepted_kwargs(f, {"a": 1, "z": 2}) == ({"a": 1, "z": 2}, [])
    assert accepted_kwargs(g, {"a": 1, "z": 2}) == ({"a": 1}, ["z"])
    assert "callback_on_step_end" in inspect.signature(RecordingPipe.__call__).parameters


def test_save_writes_video_and_manifest(generator, tmp_path):
    res = generator.generate(GenerationRequest(prompt="a cat", seed=9))
    out = res.save(tmp_path / "clip.mp4")
    reader = imageio.get_reader(str(out))
    assert reader.count_frames() == 9
    assert reader.get_meta_data()["fps"] == pytest.approx(8)
    reader.close()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["seed"] == 9 and meta["model"] == "tiny-wan" and meta["num_frames"] == 9
