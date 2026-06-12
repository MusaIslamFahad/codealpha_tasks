"""
audio_utils.py — Audio processing utilities.
Handles waveform visualisation, spectrogram generation,
audio metadata extraction, and silence trimming.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import librosa
    import librosa.display
    import soundfile as sf
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa/soundfile not installed — audio analysis disabled.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.warning("matplotlib not installed — visualisation disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_info(audio_path: str) -> dict:
    """
    Return a dict of useful audio properties.
    Falls back gracefully if librosa is absent.
    """
    if not _LIBROSA_OK:
        size_mb = os.path.getsize(audio_path) / 1_048_576
        return {"file_size_mb": round(size_mb, 2)}

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = float(np.sqrt(np.mean(y ** 2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        size_mb = os.path.getsize(audio_path) / 1_048_576

        return {
            "duration_s":      round(float(duration), 2),
            "sample_rate_hz":  int(sr),
            "estimated_bpm":   round(float(tempo), 1),
            "rms_loudness":    round(rms, 4),
            "zero_cross_rate": round(zcr, 4),
            "file_size_mb":    round(size_mb, 2),
        }
    except Exception as exc:
        logger.error("get_audio_info failed: %s", exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — deep navy / violet theme
_BG      = "#0d0f1a"
_PANEL   = "#12152a"
_ACCENT  = "#8b5cf6"    # violet
_ACCENT2 = "#06b6d4"    # cyan
_TEXT    = "#e2e8f0"
_SUBTEXT = "#94a3b8"


def generate_visualization(audio_path: str) -> Optional[str]:
    """
    Produce a 3-panel audio visualisation (waveform + mel spectrogram + chromagram)
    and save it as a PNG alongside the audio file.

    Returns the PNG path on success, or None if dependencies are missing.
    """
    if not (_LIBROSA_OK and _MPL_OK):
        logger.warning("Cannot generate visualisation — librosa or matplotlib missing.")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        fig = plt.figure(figsize=(14, 8), facecolor=_BG)
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                                left=0.07, right=0.97, top=0.88, bottom=0.08)

        # ── Panel 1 : Waveform ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        _style_axes(ax1)
        t = np.linspace(0, len(y) / sr, num=len(y))
        ax1.fill_between(t, y, alpha=0.7, color=_ACCENT, linewidth=0)
        ax1.plot(t, y, color=_ACCENT, linewidth=0.4, alpha=0.9)
        ax1.axhline(0, color=_SUBTEXT, linewidth=0.4, linestyle="--", alpha=0.5)
        ax1.set_ylabel("Amplitude", color=_SUBTEXT, fontsize=9)
        ax1.set_title("Waveform", color=_TEXT, fontsize=10, pad=6, loc="left")
        ax1.set_xlim(0, t[-1])

        # ── Panel 2 : Mel Spectrogram ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        _style_axes(ax2)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(
            mel_db, sr=sr, x_axis="time", y_axis="mel",
            ax=ax2, cmap="magma",
        )
        ax2.set_ylabel("Hz (Mel)", color=_SUBTEXT, fontsize=9)
        ax2.set_title("Mel Spectrogram", color=_TEXT, fontsize=10, pad=6, loc="left")
        cb = fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.01)
        cb.ax.yaxis.set_tick_params(color=_SUBTEXT, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=_SUBTEXT)
        cb.outline.set_edgecolor(_PANEL)

        # ── Panel 3 : Chromagram ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        _style_axes(ax3)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        img2 = librosa.display.specshow(
            chroma, sr=sr, x_axis="time", y_axis="chroma",
            ax=ax3, cmap="viridis",
        )
        ax3.set_ylabel("Pitch Class", color=_SUBTEXT, fontsize=9)
        ax3.set_xlabel("Time (s)", color=_SUBTEXT, fontsize=9)
        ax3.set_title("Chromagram", color=_TEXT, fontsize=10, pad=6, loc="left")
        cb2 = fig.colorbar(img2, ax=ax3, pad=0.01)
        cb2.ax.yaxis.set_tick_params(color=_SUBTEXT, labelsize=7)
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color=_SUBTEXT)
        cb2.outline.set_edgecolor(_PANEL)

        # ── Title bar ─────────────────────────────────────────────────────
        stem = Path(audio_path).stem
        fig.suptitle(
            f"🎵  Audio Analysis  ·  {stem}",
            color=_TEXT, fontsize=12, fontweight="bold", y=0.97,
        )

        out_png = audio_path.replace(".wav", "_analysis.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        return out_png

    except Exception as exc:
        logger.error("generate_visualization failed: %s", exc)
        return None


def _style_axes(ax) -> None:
    """Apply dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3561")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=_SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(_SUBTEXT)
    ax.yaxis.label.set_color(_SUBTEXT)
    ax.grid(color="#1e2340", linewidth=0.5, alpha=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Editing helpers
# ─────────────────────────────────────────────────────────────────────────────

def trim_silence(audio_path: str, top_db: int = 35) -> str:
    """Trim leading / trailing silence in-place. Returns the same path."""
    if not _LIBROSA_OK:
        return audio_path
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        sf.write(audio_path, y_trimmed, sr)
        logger.info("Silence trimmed: %s", audio_path)
    except Exception as exc:
        logger.error("trim_silence failed: %s", exc)
    return audio_path


def normalize_audio(audio_path: str) -> str:
    """Peak-normalize audio to –1 dBFS in-place. Returns the same path."""
    if not _LIBROSA_OK:
        return audio_path
    try:
        y, sr = librosa.load(audio_path, sr=None)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.891  # ≈ –1 dBFS
        sf.write(audio_path, y, sr)
        logger.info("Audio normalized: %s", audio_path)
    except Exception as exc:
        logger.error("normalize_audio failed: %s", exc)
    return audio_path
