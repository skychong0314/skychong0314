import diffusers
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from openreel.models import MODELS, get_model
from openreel.prompting import (
    CAMERA_MOVES,
    IMAGE_SYSTEM_SUFFIX,
    QUALITY_SUFFIX,
    STYLES,
    SYSTEM_PROMPT,
    LLMRewriter,
    apply_cast,
    compose_prompt,
)

from .conftest import tiny_spec, tiny_tokenizer


@pytest.mark.parametrize("spec", MODELS.values(), ids=list(MODELS))
def test_registry_entries_are_consistent(spec):
    for mode in spec.modes:
        assert hasattr(diffusers, spec.pipeline_for(mode)), spec.pipeline_for(mode)
        assert spec.repo_for(mode).count("/") == 1
    assert spec.snap_size(spec.width, spec.height) == (spec.width, spec.height)
    assert spec.snap_frames(spec.num_frames) == spec.num_frames
    assert hasattr(torch, spec.dtype)


def test_frame_and_size_snapping():
    spec = tiny_spec(frame_multiple=4, size_multiple=16)
    assert spec.snap_frames(81) == 81
    assert spec.snap_frames(80) == 81
    assert spec.snap_frames(3) == 5
    assert spec.frames_for(5, 16) == 81
    assert spec.frames_for(2, 8) == 17
    assert spec.snap_size(840, 470) == (832, 464)
    assert get_model("ltx-video").frames_for(5) == 121


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        get_model("seedance")


def test_compose_prompt_order_and_punctuation():
    out = compose_prompt("a fox in snow", camera="dolly-in", style="cinematic", extra="golden hour light")
    assert out.startswith("a fox in snow. " + CAMERA_MOVES["dolly-in"])
    assert out.index("golden hour light.") < out.index(STYLES["cinematic"])
    assert out.endswith(QUALITY_SUFFIX)
    assert compose_prompt("Hello!", quality=False) == "Hello!"


def test_compose_prompt_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown style 'pixar'"):
        compose_prompt("x", style="pixar")
    with pytest.raises(ValueError, match="camera move"):
        compose_prompt("x", camera="spin")
    with pytest.raises(ValueError, match="empty"):
        compose_prompt("   ")


def test_apply_cast_first_mention_and_case():
    cast = {"Mira": "silver hair, red raincoat", "Rose": "an old man"}
    out = apply_cast("Mira hands a rose to Rose. Mira smiles.", cast)
    assert out == "Mira (silver hair, red raincoat) hands a rose to Rose (an old man). Mira smiles."
    assert apply_cast("an empty street", cast) == "an empty street"
    assert apply_cast("Miranda waves", cast) == "Miranda waves"


@pytest.fixture(scope="module")
def tiny_llm():
    tok = tiny_tokenizer()
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, pad_token_id=0, eos_token_id=1)
    return LlamaForCausalLM(cfg).eval(), tok


def test_llm_rewriter_runs_generate(tiny_llm):
    model, tok = tiny_llm
    rw = LLMRewriter(model=model, tokenizer=tok, max_new_tokens=6)
    out = rw.rewrite("a fox walks in snow")
    assert isinstance(out, str) and out.strip()
    msgs = rw.messages("a fox", image_to_video=True)
    assert msgs[0]["content"] == SYSTEM_PROMPT + IMAGE_SYSTEM_SUFFIX
    assert msgs[1] == {"role": "user", "content": "a fox"}
    rw.unload()
    assert rw.model is None and rw.tokenizer is None
