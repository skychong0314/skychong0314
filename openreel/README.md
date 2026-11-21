# openreel

Seedance-style AI video generation built entirely on open-weight models.

Seedance (ByteDance) runs only on ByteDance's servers, so its model cannot be
downloaded or reverse-engineered. What *is* public is how it works: its
technical report describes the architecture and the tricks around it. openreel
rebuilds the same feature set on top of open models you can run on your own
GPU:

| Seedance feature (from its public tech report and product) | How openreel does it |
| --- | --- |
| Video VAE + diffusion transformer with separate spatial and temporal layers | Wan 2.x, LTX and HunyuanVideo use the same VAE + diffusion-transformer recipe; openreel drives them through 🤗 diffusers |
| One model for both text-to-video and image-to-video | `wan2.2-5b` is a single checkpoint for both; openreel shares its weights between the two pipelines instead of loading them twice |
| Multi-shot videos with consistent characters | **Storyboards**: a cast sheet, a shared style and seed lineage, last-frame continuity between shots, and optional reference images |
| An LLM rewrites short prompts into dense captions | Camera and style presets, plus an optional local open LLM rewriter (Qwen2.5) |
| First- and last-frame control | `--last-image` on `wan2.2-a14b` |
| Synchronized audio (newer Seedance versions) | `ltx-2` generates audio with the video and openreel muxes it into the MP4 |
| RLHF and distillation for quality and speed | Not reproduced (that needs training); `ltx-video` is the fast option |

`openreel inspect` covers the "reverse engineering" part for open models: it
reads the tensor table at the start of each `.safetensors` file (over HTTP range
requests for Hugging Face repos, so nothing large is downloaded) and reports
the real architecture: parameters per component, number of transformer
blocks, hidden sizes.

## Models

| key | model | modes | default output | VRAM (approx.) |
| --- | --- | --- | --- | --- |
| `wan2.1-1.3b` | Wan 2.1 T2V 1.3B | T2V | 832×480, 16 fps | 8 GB |
| `wan2.2-5b` (default) | Wan 2.2 TI2V 5B | T2V + I2V | 1280×704, 24 fps | 24 GB |
| `wan2.2-a14b` | Wan 2.2 A14B MoE | T2V + I2V, first+last frame | 1280×720, 16 fps | 48–80 GB |
| `ltx-video` | LTX-Video | T2V + I2V | 704×480, 24 fps | 12 GB |
| `ltx-2` | LTX-2 | T2V + I2V **with audio** | 768×512, 24 fps | 48 GB |
| `hunyuan-1.5` | HunyuanVideo 1.5 480p | T2V + I2V | 848×480, 24 fps | 24 GB |

VRAM figures assume the default `--offload model`. `--offload sequential` fits
bigger models on smaller cards at a large speed cost. Each model has its own
license (Wan is Apache-2.0; LTX and HunyuanVideo ship their own community
licenses with conditions), so check the model card before commercial use.

## Install

Needs Python 3.10+, an NVIDIA GPU (Apple Silicon works for the small models,
slowly) and `ffmpeg`-capable `imageio` (installed automatically).

```bash
cd openreel
python -m venv .venv
source .venv/bin/activate
pip install torch            # Linux: includes CUDA. Other platforms: see pytorch.org/get-started
pip install -e ".[app]"
```

Model weights download from Hugging Face on first use (5–80 GB depending on the model).

## Use it

**Web UI**

```bash
openreel app
```

Open http://127.0.0.1:7860. It has a *Generate* tab (prompt, optional first/last
frame, model, style, camera, duration, seed) and a *Storyboard* tab.

**Text to video**

```bash
openreel generate "A red fox trots through fresh snow at dawn" --style cinematic --camera tracking -d 5
```

**Image to video**

```bash
openreel generate "She turns and smiles at the camera" --image portrait.jpg
```

**First and last frame**

```bash
openreel generate "The flower blooms" -m wan2.2-a14b --image bud.jpg --last-image bloom.jpg
```

**With audio**

```bash
openreel generate "Waves crash on rocks at sunset, seagulls calling" -m ltx-2
```

**Rewrite the prompt with a local LLM first**

```bash
openreel generate "cat astronaut" --enhance llm
```

Every video is saved with a `.json` file beside it recording the final prompt,
seed and settings, so any result can be regenerated exactly. Add `--dry-run` to
see the final prompt and resolved settings without generating. `openreel presets`
lists the styles and camera moves, and `openreel models` lists the models.

## Multi-shot storyboards

```bash
openreel story examples/rainy_night.yaml -o rainy_night.mp4 --keep-shots
```

```yaml
title: Rainy night
model: wan2.2-5b
style: cinematic
seed: 42
cast:
  Mira: a young woman with short silver hair and a red raincoat
shots:
  - prompt: Mira walks through a neon-lit alley at night, rain falling
    duration: 4
    camera: tracking
  - prompt: Mira stops and looks up into the rain
    duration: 3
    transition: continue   # starts from the previous shot's last frame
  - prompt: close-up of Mira smiling, raindrops on her face
    duration: 3
    camera: dolly-in
```

The same storyboard can be written in the compact script format the web UI
uses (`examples/rainy_night.txt`):

```text
style: cinematic
cast: Mira = a young woman with short silver hair and a red raincoat
[4s] Mira walks through a neon-lit alley at night, rain falling | tracking
[+3s] Mira stops and looks up into the rain
[3s] close-up of Mira smiling, raindrops on her face | dolly-in
```

How it keeps shots consistent:

- **Cast sheet.** Each character's first mention in a shot becomes
  `Mira (a young woman with short silver hair and a red raincoat)`. Video models
  don't know names, so repeating the exact same description is what keeps
  someone's look stable from shot to shot.
- **Continuity.** `transition: continue` (or `[+3s]`) feeds the previous shot's
  last frame into image-to-video, so the action carries on with no cut. The
  duplicate frame is dropped when the shots are joined.
- **Reference images.** `image: mira.png` on a shot starts it from your own
  picture, which is the strongest way to lock in a character.
- **Seeds.** Shot *n* uses `seed + n`, and the manifest records every shot's
  final prompt and seed.

Continuing a shot or using a reference image needs a model with image-to-video
support. openreel checks this before loading anything.

## Inspect a model's architecture

```bash
openreel inspect wan2.2-5b
openreel inspect Lightricks/LTX-Video --json
openreel inspect ./my-downloaded-model/transformer
```

For each component (transformer, text encoder, VAE, image encoder) it prints the
class, parameter count, storage dtype and size, how many times each block
repeats (`blocks×30` means 30 transformer layers), and the key config values
(attention heads, head size, latent channels, patch size). It works on a local
diffusers folder, a single `.safetensors` file, a Hugging Face repo id, or any
model key above.

## Python API

```python
from openreel import GenerationRequest, VideoGenerator, get_model, compose_prompt

gen = VideoGenerator(get_model("wan2.2-5b"))
result = gen.generate(GenerationRequest(
    prompt=compose_prompt("A red fox trots through fresh snow", style="cinematic", camera="tracking"),
    duration=5,
    seed=42,
))
result.save("fox.mp4")
```

## Tests

```bash
pip install -e ".[app,test]"
pytest
```

The tests build tiny random-weight versions of the real Wan components: the
same diffusers and transformers classes, shrunk to a few thousand parameters.
They run the actual pipelines (text-to-video, image-to-video, first+last frame,
and the Wan 2.2 TI2V weight-sharing path) on CPU in a few seconds, with no
downloads. They also cover storyboard chaining and audio sync, video and audio
encoding, argument filtering for pipelines with different signatures, the CLI,
the web UI handlers and `inspect`.

## Layout

```text
src/openreel/
  models.py         model registry: repos, pipeline classes, defaults
  generator.py      loads pipelines, memory strategy, runs one generation
  prompting.py      style/camera presets, cast substitution, LLM rewriter
  storyboard.py     multi-shot parsing, planning, rendering, stitching
  media.py          frame/image/audio helpers, MP4 + audio encoding
  inspect_model.py  safetensors-header architecture inspection
  cli.py            the openreel command
  app.py            Gradio web UI
examples/           sample storyboards (YAML and script format)
tests/
```
