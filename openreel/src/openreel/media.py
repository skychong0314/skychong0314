"""Frame, image, audio and video-file helpers.

Frames are passed around as float32 numpy arrays shaped ``(T, H, W, 3)`` with
values in ``[0, 1]``, which is what diffusers returns for ``output_type="np"``.
Audio is ``(channels, samples)`` float in ``[-1, 1]``.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def to_float_frames(video: Any) -> np.ndarray:
    """Normalise a pipeline's video output to ``(T, H, W, 3)`` float32 in [0, 1]."""
    if isinstance(video, (list, tuple)):
        if video and isinstance(video[0], Image.Image):
            video = np.stack([np.asarray(f.convert("RGB")) for f in video])
        else:
            video = np.stack([np.asarray(f) for f in video])
    elif hasattr(video, "detach"):  # torch.Tensor
        video = video.detach().float().cpu().numpy()
    arr = np.asarray(video)
    if arr.ndim == 5:  # (B, T, H, W, C): keep the first video
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected a 4D video array, got shape {arr.shape}")
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[1] in (1, 3, 4):  # (T, C, H, W)
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.clip(arr, 0.0, 1.0)


def to_uint8(frames: np.ndarray) -> np.ndarray:
    return (np.clip(frames, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop and resize an image to exactly ``width x height``."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size == (width, height):
        return image
    return ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def frame_to_image(frames: np.ndarray, index: int = -1) -> Image.Image:
    return Image.fromarray(to_uint8(frames[index]))


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def to_audio_array(audio: Any) -> np.ndarray:
    """Normalise audio to ``(channels, samples)`` float32."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3:  # (B, C, S)
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected audio shaped (channels, samples), got {arr.shape}")
    if arr.shape[0] > arr.shape[1]:  # (samples, channels)
        arr = arr.T
    return arr


def fit_audio(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Trim or zero-pad audio to exactly ``num_samples``."""
    if audio.shape[1] >= num_samples:
        return audio[:, :num_samples]
    pad = np.zeros((audio.shape[0], num_samples - audio.shape[1]), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=1)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.T.tobytes())
    return path


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_frames(path: Path, frames: np.ndarray, fps: float, quality: float) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=quality,
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_log_level="error",
    )
    try:
        for frame in to_uint8(frames):
            writer.append_data(frame)
    finally:
        writer.close()


def save_video(
    path: str | Path,
    frames: np.ndarray,
    fps: float,
    *,
    audio: np.ndarray | None = None,
    sample_rate: int | None = None,
    quality: float = 8.0,
) -> Path:
    """Encode frames (and optional audio) to an H.264 MP4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio is None:
        _write_frames(path, frames, fps, quality)
        return path
    if not sample_rate:
        raise ValueError("sample_rate is required when audio is given")
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "video.mp4"
        wav = Path(tmp) / "audio.wav"
        _write_frames(silent, frames, fps, quality)
        write_wav(wav, to_audio_array(audio), sample_rate)
        subprocess.run(
            [
                _ffmpeg(), "-y", "-loglevel", "error",
                "-i", str(silent), "-i", str(wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", "apad", "-shortest",
                str(path),
            ],
            check=True,
        )
    return path
