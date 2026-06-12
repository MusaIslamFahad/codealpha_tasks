"""
generator.py — Core music generation engine.
Wraps Meta's MusicGen with intelligent model management,
device detection, and clean generation API.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation) ────────────────────────────
try:
    import torchaudio
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    _AUDIOCRAFT_OK = True
except ImportError:
    _AUDIOCRAFT_OK = False
    logger.warning(
        "audiocraft is not installed.\n"
        "Install it with:  pip install audiocraft\n"
        "Or run:           pip install -r requirements.txt"
    )

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CFG_COEF,
    DEFAULT_DURATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    OUTPUT_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
class MusicGenerator:
    """
    Singleton-friendly wrapper around Meta's MusicGen.

    Features
    --------
    - Lazy model loading with in-memory caching (avoids redundant downloads)
    - Automatic device selection: CUDA → MPS → CPU
    - Text-to-music and melody-conditioned generation
    - Structured generation metadata returned alongside audio path
    """

    def __init__(self) -> None:
        self.model: Optional[object] = None
        self.current_model_key: Optional[str] = None
        self.device = self._best_device()
        logger.info("MusicGenerator ready. Device: %s", self.device)

    # ── Device helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _best_device() -> str:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", name)
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected")
            return "mps"
        logger.info("No GPU found — using CPU (generation will be slow)")
        return "cpu"

    def gpu_memory_mb(self) -> float:
        """Return allocated GPU memory in MB (0 on CPU)."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e6
        return 0.0

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(self, model_key: str) -> None:
        """Load *model_key* if it differs from the currently loaded model."""
        if not _AUDIOCRAFT_OK:
            raise RuntimeError(
                "audiocraft is required. Install it with:\n"
                "  pip install audiocraft"
            )
        if self.current_model_key == model_key and self.model is not None:
            return  # already loaded — no-op

        hf_name = AVAILABLE_MODELS[model_key]
        logger.info("Loading %s …", hf_name)

        # Free previous model
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.model = MusicGen.get_pretrained(hf_name)
        self.current_model_key = model_key
        logger.info("Model loaded: %s (%.0f MB GPU)", hf_name, self.gpu_memory_mb())

    def unload_model(self) -> None:
        """Release the currently loaded model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_key = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model_key: str,
        duration: int = DEFAULT_DURATION,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        cfg_coef: float = DEFAULT_CFG_COEF,
        melody_audio_path: Optional[str] = None,
    ) -> tuple[str, dict]:
        """
        Generate music from *prompt* and return (wav_path, metadata).

        Parameters
        ----------
        prompt            : Natural language description of the music.
        model_key         : Key in AVAILABLE_MODELS.
        duration          : Output duration in seconds.
        temperature       : Sampling temperature (higher → more creative).
        top_k             : Top-K sampling parameter.
        top_p             : Nucleus sampling probability (0 = disabled).
        cfg_coef          : Classifier-free guidance scale.
        melody_audio_path : Path to a melody WAV/MP3 for chroma conditioning
                            (only used with the Melody model variant).

        Returns
        -------
        wav_path   : Absolute path to the generated .wav file.
        metadata   : Dict with generation details for display/logging.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        self.load_model(model_key)

        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )

        t0 = time.perf_counter()

        use_melody = (
            melody_audio_path is not None
            and os.path.isfile(melody_audio_path)
            and "Melody" in model_key
        )

        if use_melody:
            logger.info("Melody-conditioned generation — %s", melody_audio_path)
            melody_wav, sr = torchaudio.load(melody_audio_path)
            if melody_wav.dim() == 1:
                melody_wav = melody_wav.unsqueeze(0)
            wav = self.model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody_wav.unsqueeze(0),
                melody_sample_rate=sr,
                progress=True,
            )
        else:
            logger.info("Text-to-music generation — %s …", prompt[:60])
            wav = self.model.generate([prompt], progress=True)

        elapsed = time.perf_counter() - t0

        # ── Write output ───────────────────────────────────────────────────
        stem = f"generated_{int(time.time())}"
        out_path = str(OUTPUT_DIR / stem)

        audio_write(
            out_path,
            wav[0].cpu(),
            self.model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )

        wav_file = f"{out_path}.wav"
        logger.info("Saved → %s  (%.1f s generation time)", wav_file, elapsed)

        metadata = {
            "model_key":         model_key,
            "model_hf":          AVAILABLE_MODELS[model_key],
            "prompt":            prompt,
            "duration_s":        duration,
            "temperature":       temperature,
            "top_k":             top_k,
            "top_p":             top_p,
            "cfg_coef":          cfg_coef,
            "melody_conditioned": use_melody,
            "generation_time_s": round(elapsed, 2),
            "output_path":       wav_file,
            "device":            self.device,
        }
        return wav_file, metadata
