"""Inspect a video model's architecture from its files, without loading it.

A safetensors file starts with an 8-byte length and a JSON header that lists
every tensor's name, dtype and shape. Reading only that header (a few hundred
KB even for a 30 GB checkpoint, fetched with HTTP range requests when the
model is on Hugging Face) reveals the real structure: parameter counts per
component, how many transformer blocks there are, hidden sizes and so on.

Works on a local diffusers folder, a single .safetensors file, a Hugging Face
repo id, or an openreel model key.
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 0.5,
}

CONFIG_KEYS = (
    "num_layers", "num_hidden_layers", "num_attention_heads", "attention_head_dim", "hidden_size",
    "d_model", "in_channels", "out_channels", "patch_size", "text_dim", "ffn_dim", "d_ff",
    "cross_attention_dim", "caption_channels", "image_dim", "z_dim", "latent_channels",
    "base_dim", "scale_factor_spatial", "scale_factor_temporal", "temperal_downsample",
    "vocab_size", "boundary_ratio", "expand_timesteps",
)

MAX_HEADER_BYTES = 100 * 1024 * 1024


def read_safetensors_header(f: Any) -> dict[str, Any]:
    """Read the tensor table from an open binary safetensors file."""
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("file is too short to be safetensors")
    (length,) = struct.unpack("<Q", raw)
    if length > MAX_HEADER_BYTES:
        raise ValueError(f"implausible safetensors header length {length}")
    header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


@dataclass
class ComponentReport:
    name: str
    class_name: str | None
    files: list[str]
    params: int
    size_bytes: int
    dtypes: dict[str, int]
    repeated_blocks: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelReport:
    source: str
    pipeline: str | None
    components: list[ComponentReport]

    @property
    def total_params(self) -> int:
        return sum(c.params for c in self.components)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "pipeline": self.pipeline, "total_params": self.total_params,
                "components": [asdict(c) for c in self.components]}


def summarize_tensors(header: dict[str, Any]) -> tuple[int, int, Counter, dict[str, int]]:
    params = 0
    size = 0.0
    dtypes: Counter = Counter()
    block_max: dict[str, int] = {}
    block_re = re.compile(r"^(.*?)\.(\d+)\.")
    for name, info in header.items():
        n = 1
        for dim in info["shape"]:
            n *= dim
        params += n
        size += n * DTYPE_BYTES.get(info["dtype"], 0)
        dtypes[info["dtype"]] += n
        m = block_re.match(name)
        if m:
            prefix, idx = m.group(1), int(m.group(2))
            block_max[prefix] = max(block_max.get(prefix, -1), idx)
    blocks = {k: v + 1 for k, v in sorted(block_max.items(), key=lambda kv: -kv[1]) if v >= 1}
    return params, int(size), dtypes, blocks


def _pick_weight_files(fs: Any, folder: str) -> list[str]:
    names = [p.rsplit("/", 1)[-1] for p in fs.ls(folder, detail=False)]
    indexes = sorted(n for n in names if n.endswith(".safetensors.index.json"))
    if indexes:
        index = min(indexes, key=lambda n: (n.count("."), len(n)))  # prefer non-variant
        with fs.open(f"{folder}/{index}", "rb") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))
    files = sorted(n for n in names if n.endswith(".safetensors"))
    plain = [n for n in files if n[: -len(".safetensors")].count(".") == 0]
    return plain or files


def _read_json(fs: Any, path: str) -> dict[str, Any] | None:
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as f:
        return json.load(f)


def inspect_component(fs: Any, folder: str, name: str, class_name: str | None = None) -> ComponentReport | None:
    files = _pick_weight_files(fs, folder)
    if not files:
        return None
    params, size, dtypes, blocks = 0, 0, Counter(), {}
    for fname in files:
        with fs.open(f"{folder}/{fname}", "rb") as f:
            header = read_safetensors_header(f)
        p, s, d, b = summarize_tensors(header)
        params += p
        size += s
        dtypes.update(d)
        for k, v in b.items():
            blocks[k] = max(blocks.get(k, 0), v)
    config = _read_json(fs, f"{folder}/config.json") or {}
    picked = {k: config[k] for k in CONFIG_KEYS if k in config}
    return ComponentReport(
        name=name,
        class_name=class_name or config.get("_class_name") or (config.get("architectures") or [None])[0],
        files=files,
        params=params,
        size_bytes=size,
        dtypes=dict(dtypes),
        repeated_blocks=dict(sorted(blocks.items(), key=lambda kv: -kv[1])[:4]),
        config=picked,
    )


def resolve_source(source: str, fs: Any = None) -> tuple[Any, str]:
    """Return an fsspec filesystem and root path for a key, folder, file or repo id."""
    from .models import MODELS

    if source in MODELS:
        source = MODELS[source].t2v_repo or MODELS[source].i2v_repo
    if fs is not None:
        return fs, source.rstrip("/")
    if os.path.exists(source):
        from fsspec.implementations.local import LocalFileSystem

        return LocalFileSystem(), os.path.abspath(source)
    from huggingface_hub import HfFileSystem

    return HfFileSystem(), source.rstrip("/")


def inspect_model(source: str, *, fs: Any = None) -> ModelReport:
    fs, root = resolve_source(source, fs)
    if root.endswith(".safetensors"):
        with fs.open(root, "rb") as f:
            header = read_safetensors_header(f)
        params, size, dtypes, blocks = summarize_tensors(header)
        comp = ComponentReport(os.path.basename(root), None, [os.path.basename(root)], params, size,
                               dict(dtypes), dict(list(blocks.items())[:4]))
        return ModelReport(source, None, [comp])

    index = _read_json(fs, f"{root}/model_index.json")
    if index is None:  # a single component folder
        comp = inspect_component(fs, root, os.path.basename(root))
        if comp is None:
            raise FileNotFoundError(f"no model_index.json or .safetensors files in {source}")
        return ModelReport(source, None, [comp])

    components = []
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or value[0] is None:
            continue
        folder = f"{root}/{name}"
        if not fs.exists(folder):
            continue
        comp = inspect_component(fs, folder, name, value[1])
        if comp is not None:
            components.append(comp)
    return ModelReport(source, index.get("_class_name"), components)


def _human(n: float, unit: str = "") -> str:
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{suffix}{unit}"
    return f"{n:.0f}{unit}"


def format_report(report: ModelReport) -> str:
    lines = [f"{report.source}" + (f"  ({report.pipeline})" if report.pipeline else "")]
    width = max((len(c.name) for c in report.components), default=0)
    for c in report.components:
        main_dtype = max(c.dtypes, key=c.dtypes.get) if c.dtypes else "?"
        lines.append(
            f"  {c.name:<{width}}  {c.class_name or '':<28} {_human(c.params):>8} params"
            f"  {main_dtype:<5} {c.size_bytes / 1e9:6.2f} GB"
        )
        details = [f"{k}×{v}" for k, v in c.repeated_blocks.items()]
        details += [f"{k}={v}" for k, v in c.config.items()]
        if details:
            lines.append(f"  {'':<{width}}  " + " · ".join(details))
    lines.append(f"  total {_human(report.total_params)} parameters")
    return "\n".join(lines)
