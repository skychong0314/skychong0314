"""Tiny random-weight Wan pipelines that run the real diffusers code on CPU.

The components use the same classes as the released checkpoints, shrunk to a
few thousand parameters, so a full denoise + VAE decode takes milliseconds and
needs no downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

from openreel.generator import VideoGenerator
from openreel.models import ModelSpec

WORDS = "a an the cat dog fox woman walks runs on in at beach snow alley night rain camera slow".split()


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")


def tiny_transformer(in_channels: int) -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=in_channels,
        out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
    ).eval()


@pytest.fixture(scope="session")
def components():
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    text_encoder = T5EncoderModel(
        T5Config(vocab_size=len(tokenizer), d_model=32, d_ff=37, num_layers=2, num_heads=4, d_kv=8)
    ).eval()
    vae = AutoencoderKLWan(
        base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1, temperal_downsample=[False, True, True]
    ).eval()
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae}


@pytest.fixture(scope="session")
def t2v_pipe(components):
    torch.manual_seed(1)
    return WanPipeline(scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0), transformer=tiny_transformer(16),
                       **components)


@pytest.fixture(scope="session")
def i2v_pipe(components):
    """Wan 2.2 A14B-style I2V: image latents + mask concatenated (16 + 16 + 4 channels)."""
    torch.manual_seed(2)
    return WanImageToVideoPipeline(scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0),
                                   transformer=tiny_transformer(36), **components)


@pytest.fixture(scope="session")
def ti2v_t2v_pipe(components):
    """Wan 2.2 TI2V-5B-style: one transformer, expand_timesteps, I2V derived via from_pipe."""
    torch.manual_seed(3)
    return WanPipeline(scheduler=FlowMatchEulerDiscreteScheduler(shift=7.0), transformer=tiny_transformer(16),
                       expand_timesteps=True, **components)


def tiny_spec(**overrides) -> ModelSpec:
    base = dict(
        key="tiny-wan", name="Tiny Wan", family="wan", summary="test model",
        t2v_repo="local/tiny", t2v_pipeline="WanPipeline",
        i2v_repo="local/tiny-i2v", i2v_pipeline="WanImageToVideoPipeline",
        width=32, height=32, num_frames=9, fps=8, steps=2, guidance=5.0, last_frame=True,
    )
    base.update(overrides)
    return ModelSpec(**base)


@pytest.fixture
def spec():
    return tiny_spec()


@pytest.fixture
def generator(spec, t2v_pipe, i2v_pipe):
    return VideoGenerator(spec, device="cpu", t2v_pipe=t2v_pipe, i2v_pipe=i2v_pipe)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((48, 64, 3)) * 255).astype("uint8"))
