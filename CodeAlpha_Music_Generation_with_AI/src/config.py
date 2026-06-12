"""
Configuration settings for AI Music Generation Studio.
Centralises all model parameters, paths, and default values.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Available MusicGen models ─────────────────────────────────────────────────
AVAILABLE_MODELS: dict[str, str] = {
    "MusicGen Small  (300M)  — Fast":          "facebook/musicgen-small",
    "MusicGen Medium (1.5B)  — Balanced":      "facebook/musicgen-medium",
    "MusicGen Large  (3.3B)  — Best Quality":  "facebook/musicgen-large",
    "MusicGen Melody (1.5B)  — Melody Input":  "facebook/musicgen-melody",
}

DEFAULT_MODEL = "MusicGen Small  (300M)  — Fast"

# ── Generation hyper-parameters ───────────────────────────────────────────────
DEFAULT_DURATION    = 10
MIN_DURATION        = 5
MAX_DURATION        = 30

DEFAULT_TEMPERATURE = 1.0
MIN_TEMPERATURE     = 0.1
MAX_TEMPERATURE     = 2.0

DEFAULT_TOP_K       = 250
MIN_TOP_K           = 0
MAX_TOP_K           = 1000

DEFAULT_TOP_P       = 0.0
MIN_TOP_P           = 0.0
MAX_TOP_P           = 1.0

DEFAULT_CFG_COEF    = 3.0
MIN_CFG_COEF        = 1.0
MAX_CFG_COEF        = 10.0

# re-export aliases used by app.py
MIN_TEMPERATURE     = 0.1
MAX_TEMPERATURE     = 2.0
MIN_TOP_K           = 0
MAX_TOP_K           = 1000
MIN_TOP_P           = 0.0
MAX_TOP_P           = 1.0

SAMPLE_RATE         = 32000   # MusicGen native sample rate

# ── Curated example prompts ───────────────────────────────────────────────────
EXAMPLE_PROMPTS: list[str] = [
    "An upbeat jazz piano trio with walking bass and brushed snare, cozy coffee-shop vibe",
    "Epic orchestral battle theme with soaring strings, brass fanfares and thunderous percussion",
    "Chill lo-fi hip hop beat with vinyl crackle, mellow Rhodes and a dusty drum loop",
    "Melancholic classical piano sonata, slow and introspective, in the style of Chopin",
    "Energetic EDM drop with supersaw leads, punching kick drum and euphoric build-up",
    "Fingerpicked acoustic guitar in open tuning, warm and folksy, autumn countryside feel",
    "Dark ambient drone with deep sub-bass rumble, eerie textures and distant whispers",
    "Funky 70s disco groove — wah guitar, slap bass, brass stabs and tambourine",
    "Gentle meditation soundscape with Tibetan singing bowls, soft flute and rain sounds",
    "Aggressive punk rock — distorted power chords, driving drums, raw electric energy",
    "Dreamy synth-wave with lush pads, arpeggiated sequences and reverb-soaked vocals",
    "Traditional Irish jig on fiddle and bodhrán, lively and danceable, 6/8 time",
]

# ── Prompt guidance tips ──────────────────────────────────────────────────────
PROMPT_TIPS: list[str] = [
    "Name specific instruments (e.g. 'upright bass', 'Rhodes piano', 'marimba')",
    "Include mood/emotion words (e.g. 'melancholic', 'euphoric', 'tense')",
    "Mention tempo or feel (e.g. '120 BPM', 'slow waltz', 'driving 4/4')",
    "Reference a genre or era (e.g. '70s soul', 'baroque', 'synthwave')",
    "Describe the setting (e.g. 'concert hall acoustics', 'lo-fi bedroom recording')",
    "Use CFG ≥ 4 for strict prompt adherence; lower values give more freedom",
    "Temperature 0.7–1.0 → coherent; 1.2–1.5 → experimental/creative",
    "For melody conditioning, upload any short audio clip and switch to the Melody model",
]
