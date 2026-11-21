import json
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from safetensors.torch import save_file

from openreel import app as app_module
from openreel.cli import main
from openreel.generator import VideoGenerator
from openreel.inspect_model import format_report, inspect_model
from openreel.models import MODELS

from .conftest import tiny_spec, tiny_transformer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def tiny_registered(monkeypatch, t2v_pipe, i2v_pipe):
    monkeypatch.setitem(MODELS, "tiny-wan", tiny_spec())

    def factory(spec, args=None):
        return VideoGenerator(spec, device="cpu", t2v_pipe=t2v_pipe, i2v_pipe=i2v_pipe)

    return factory


# -- CLI -----------------------------------------------------------------------

def test_models_and_presets(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in MODELS) and "default: wan2.2-5b" in out
    assert main(["presets"]) == 0
    assert "dolly-in" in capsys.readouterr().out


def test_generate_dry_run_uses_real_defaults(capsys):
    assert main(["generate", "a fox in snow", "-m", "wan2.2-5b", "-s", "cinematic", "-d", "5",
                 "--seed", "1", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["width"], plan["height"], plan["num_frames"], plan["fps"]) == (1280, 704, 121, 24)
    assert plan["prompt"].startswith("a fox in snow.") and "35mm" in plan["prompt"]


def test_generate_writes_files(tiny_registered, tmp_path, image, capsys):
    out = tmp_path / "fox.mp4"
    assert main(["generate", "a fox", "-m", "tiny-wan", "--seed", "4", "-o", str(out)],
                factory=tiny_registered) == 0
    assert out.exists() and json.loads(out.with_suffix(".json").read_text())["seed"] == 4
    img = tmp_path / "start.png"
    image.save(img)
    out2 = tmp_path / "i2v.mp4"
    assert main(["generate", "it moves", "-m", "tiny-wan", "-i", str(img), "--raw", "-o", str(out2)],
                factory=tiny_registered) == 0
    meta = json.loads(out2.with_suffix(".json").read_text())
    assert meta["mode"] == "i2v" and meta["prompt"] == "it moves"
    assert "saved" in capsys.readouterr().out


def test_generate_reports_unsupported_mode(tmp_path, image, capsys):
    img = tmp_path / "start.png"
    image.save(img)
    assert main(["generate", "x", "-m", "wan2.1-1.3b", "-i", str(img), "--dry-run"]) == 2
    assert "cannot start from an image" in capsys.readouterr().err


def test_story_dry_run_and_render(tiny_registered, tmp_path, capsys):
    assert main(["story", str(EXAMPLES / "rainy_night.txt"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 shots" in out and "[continue," in out and "wan2.2-5b" in out

    script = tmp_path / "s.txt"
    script.write_text("cast: Fox = orange fur\n[1s] Fox in snow\n[+1s] Fox runs | tracking\n")
    video = tmp_path / "s.mp4"
    assert main(["story", str(script), "-m", "tiny-wan", "--seed", "3", "-o", str(video), "--keep-shots"],
                factory=tiny_registered) == 0
    manifest = json.loads(video.with_suffix(".json").read_text())
    assert [s["seed"] for s in manifest["shots"]] == [3, 4]
    assert manifest["num_frames"] == 9 + 8
    assert (tmp_path / "s_shot02.mp4").exists()


# -- inspect ---------------------------------------------------------------------

def _fake_repo(root: Path) -> int:
    tr = tiny_transformer(16)
    state = {k: v.contiguous() for k, v in tr.state_dict().items()}
    (root / "transformer").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer_2": [None, None],
    }))
    (root / "transformer" / "config.json").write_text(json.dumps(
        {"_class_name": "WanTransformer3DModel", **{k: v for k, v in tr.config.items() if not k.startswith("_")}}
    ))
    keys = sorted(state)
    half = len(keys) // 2
    shards = {"m-00001-of-00002.safetensors": keys[:half], "m-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for fname, names in shards.items():
        save_file({k: state[k] for k in names}, str(root / "transformer" / fname))
        weight_map.update({k: fname for k in names})
    (root / "transformer" / "m.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    # A variant copy that must not be double counted.
    save_file(state, str(root / "transformer" / "m.fp16.safetensors"))
    return sum(v.numel() for v in state.values())


def test_inspect_local_repo(tmp_path):
    expected = _fake_repo(tmp_path / "repo")
    report = inspect_model(str(tmp_path / "repo"))
    assert report.pipeline == "WanPipeline"
    [comp] = report.components
    assert comp.name == "transformer" and comp.class_name == "WanTransformer3DModel"
    assert comp.params == expected
    assert comp.repeated_blocks["blocks"] == 2
    assert comp.config["num_layers"] == 2 and comp.config["in_channels"] == 16
    assert len(comp.files) == 2
    text = format_report(report)
    assert "transformer" in text and "blocks×2" in text


def test_inspect_through_fsspec_like_a_remote_repo(tmp_path):
    expected = _fake_repo(tmp_path / "org" / "model")
    report = inspect_model(str(tmp_path / "org" / "model"), fs=LocalFileSystem())
    assert report.total_params == expected


def test_inspect_single_file_and_cli(tmp_path, capsys):
    _fake_repo(tmp_path / "repo")
    single = tmp_path / "repo" / "transformer" / "m.fp16.safetensors"
    report = inspect_model(str(single))
    assert report.components[0].repeated_blocks["blocks"] == 2
    assert main(["inspect", str(tmp_path / "repo"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pipeline"] == "WanPipeline"
    assert main(["inspect", str(tmp_path)]) == 2


# -- web UI ----------------------------------------------------------------------

def test_app_builds_and_handlers_run(tiny_registered, tmp_path, monkeypatch, image):
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path / "outputs")
    demo = app_module.build_app(default_model="tiny-wan", factory=tiny_registered)
    assert demo is not None

    cache = app_module.GeneratorCache(tiny_registered)
    steps = []
    path, meta = app_module.run_generate(cache, "a fox", image=image, model_key="tiny-wan", style="anime",
                                         camera="(none)", seed=11, progress=lambda f, m: steps.append(m))
    assert Path(path).exists() and meta["seed"] == 11 and meta["mode"] == "i2v"
    assert steps == ["Denoising step 1/2", "Denoising step 2/2"]

    path, manifest = app_module.run_story(cache, "[1s] a fox\n[+1s] it runs", model_key="tiny-wan", seed=2)
    assert Path(path).exists() and len(manifest["shots"]) == 2
    with pytest.raises(ValueError, match="prompt"):
        app_module.run_generate(cache, "  ", model_key="tiny-wan")
