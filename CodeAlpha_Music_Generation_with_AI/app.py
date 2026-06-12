"""
app.py — AI Music Generation Studio
Entry point for the Gradio web application.

Run locally :  python app.py
HuggingFace :  set app_file: app.py in README.md frontmatter
"""

from __future__ import annotations

import logging
import os

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_CFG_COEF,
    DEFAULT_DURATION,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    EXAMPLE_PROMPTS,
    MAX_CFG_COEF,
    MAX_DURATION,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MAX_TOP_P,
    MIN_CFG_COEF,
    MIN_DURATION,
    MIN_TEMPERATURE,
    MIN_TOP_K,
    MIN_TOP_P,
    PROMPT_TIPS,
)
from src.generator import MusicGenerator
from src.audio_utils import generate_visualization, get_audio_info

# ── Global generator (lazy model loading) ────────────────────────────────────
generator = MusicGenerator()

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #080b14;
    --surface:   #0f1220;
    --surface2:  #161a2e;
    --border:    #1e2340;
    --accent:    #8b5cf6;
    --accent2:   #06b6d4;
    --accent3:   #f472b6;
    --text:      #e2e8f0;
    --subtext:   #94a3b8;
    --radius:    12px;
    --glow:      0 0 24px rgba(139,92,246,0.25);
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(160deg, #0d0f1a 0%, #120e2a 50%, #0d1a1f 100%);
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 60% at 50% -10%,
        rgba(139,92,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    color: var(--subtext);
    font-size: 1rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(139,92,246,0.18);
    border: 1px solid rgba(139,92,246,0.4);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

/* ── Panels ── */
.gr-block, .gr-box, .gr-group,
.gradio-group, .panel-wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--subtext) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ── Inputs ── */
input, textarea, select, .gr-text-input, .gr-dropdown {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
    outline: none !important;
}
label span {
    color: var(--subtext) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Sliders ── */
input[type=range] { accent-color: var(--accent) !important; }

/* ── Primary button ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: var(--glow) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
.gr-button-primary:hover, button.primary:hover {
    filter: brightness(1.12) !important;
    box-shadow: 0 0 32px rgba(139,92,246,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
.gr-button-secondary, button.secondary {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Audio player ── */
.gr-audio audio {
    width: 100% !important;
    border-radius: 8px !important;
    accent-color: var(--accent) !important;
}

/* ── Info cards ── */
.info-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--subtext);
}
.info-card strong { color: var(--text); }
.info-card .success { color: #34d399; }
.info-card .warn    { color: #fbbf24; }

/* ── Stat pills ── */
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; }
.stat-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    color: var(--subtext);
}
.stat-pill b { color: var(--accent2); }

/* ── Example prompts ── */
.gr-examples table { background: var(--surface) !important; }
.gr-examples td, .gr-examples th {
    border-color: var(--border) !important;
    color: var(--subtext) !important;
    font-size: 0.83rem !important;
}
.gr-examples tr:hover td { background: var(--surface2) !important; color: var(--text) !important; }

/* ── Tips list ── */
.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.tip-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    font-size: 0.83rem;
    color: var(--subtext);
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}
.tip-card::before { content: '✦'; color: var(--accent); flex-shrink: 0; }

/* ── Accordion ── */
details summary {
    color: var(--subtext) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
details[open] summary { color: var(--accent) !important; }

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    padding: 1.5rem 1rem;
    border-top: 1px solid var(--border);
    color: var(--subtext);
    font-size: 0.8rem;
    margin-top: 2rem;
}
.footer-wrap a { color: var(--accent); text-decoration: none; }
.footer-wrap a:hover { text-decoration: underline; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Generation handler
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(
    prompt: str,
    model_key: str,
    duration: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
    melody_audio,
    progress=gr.Progress(track_tqdm=True),
):
    """Gradio callback: validate → generate → visualise → return."""
    if not prompt.strip():
        gr.Warning("Please enter a music description before generating.")
        return None, None, "⚠️  Enter a description above and click **Generate Music**."

    try:
        progress(0.05, desc="Initialising model …")
        melody_path = melody_audio if isinstance(melody_audio, str) else None

        progress(0.15, desc="Loading model weights …")
        audio_path, meta = generator.generate(
            prompt=prompt,
            model_key=model_key,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
            melody_audio_path=melody_path,
        )

        progress(0.85, desc="Analysing audio …")
        info    = get_audio_info(audio_path)
        vis_png = generate_visualization(audio_path)

        progress(1.0, desc="Done!")

        model_short = meta["model_hf"].split("/")[-1]
        conditioned = "✅ Yes" if meta["melody_conditioned"] else "No"

        md = f"""
### ✅ Generation complete

| Property | Value |
|---|---|
| **Model** | `{model_short}` |
| **Prompt** | {prompt[:90]}{'…' if len(prompt) > 90 else ''} |
| **Duration** | {meta['duration_s']} s |
| **Estimated BPM** | {info.get('estimated_bpm', '—')} |
| **File size** | {info.get('file_size_mb', '—')} MB |
| **Generation time** | {meta['generation_time_s']} s |
| **Device** | `{meta['device']}` |
| **Melody-conditioned** | {conditioned} |
| **Temperature / Top-K / CFG** | {temperature} / {top_k} / {cfg_coef} |
""".strip()

        return audio_path, vis_png, md

    except Exception as exc:
        logger.exception("Generation failed")
        err = (
            f"❌ **Error:** {exc}\n\n"
            "_Tip: if you hit an out-of-memory error, choose a smaller model "
            "or reduce the duration._"
        )
        return None, None, err


# ─────────────────────────────────────────────────────────────────────────────
# Interface builder
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="AI Music Generation Studio",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:

        # ── Hero ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="hero-wrap">
          <div class="badge">AI · Music · Generation</div>
          <h1 class="hero-title">Music Generation <span>Studio</span></h1>
          <p class="hero-sub">Powered by Meta's MusicGen · Text-to-music · Melody conditioning · Audio analysis</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — GENERATE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🎼  Generate"):
                with gr.Row(equal_height=False):

                    # ── Left column — controls ─────────────────────────
                    with gr.Column(scale=5, min_width=340):

                        prompt_box = gr.Textbox(
                            label="Music Description",
                            placeholder=(
                                "Describe the music you want…\n"
                                "e.g. Upbeat jazz piano trio with walking bass and "
                                "brushed snare — cozy coffee-shop vibe"
                            ),
                            lines=4,
                            max_lines=10,
                            show_label=True,
                        )

                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=list(AVAILABLE_MODELS.keys()),
                                value=DEFAULT_MODEL,
                                label="Model",
                                info="Larger = better quality, slower",
                                scale=3,
                            )
                            duration_sl = gr.Slider(
                                minimum=MIN_DURATION,
                                maximum=MAX_DURATION,
                                value=DEFAULT_DURATION,
                                step=1,
                                label="Duration (s)",
                                scale=2,
                            )

                        with gr.Accordion("⚙️  Advanced Parameters", open=False):
                            gr.HTML("""
                            <p style='font-size:0.8rem;color:#64748b;margin:0 0 0.75rem'>
                            Defaults work well for most prompts. Adjust only if you want
                            finer creative control.
                            </p>""")
                            with gr.Row():
                                temp_sl = gr.Slider(
                                    minimum=MIN_TEMPERATURE, maximum=MAX_TEMPERATURE,
                                    value=DEFAULT_TEMPERATURE, step=0.05,
                                    label="🌡️ Temperature",
                                    info="Higher → more creative / unpredictable",
                                )
                                cfg_sl = gr.Slider(
                                    minimum=MIN_CFG_COEF, maximum=MAX_CFG_COEF,
                                    value=DEFAULT_CFG_COEF, step=0.5,
                                    label="🎯 CFG Coefficient",
                                    info="Higher → closer adherence to prompt",
                                )
                            with gr.Row():
                                topk_sl = gr.Slider(
                                    minimum=MIN_TOP_K, maximum=MAX_TOP_K,
                                    value=DEFAULT_TOP_K, step=10,
                                    label="🔝 Top-K",
                                    info="0 = disabled",
                                )
                                topp_sl = gr.Slider(
                                    minimum=MIN_TOP_P, maximum=MAX_TOP_P,
                                    value=DEFAULT_TOP_P, step=0.05,
                                    label="🎲 Top-P (Nucleus)",
                                    info="0 = disabled (use Top-K instead)",
                                )

                        with gr.Accordion("🎼  Melody Conditioning", open=False):
                            gr.HTML("""
                            <p style='font-size:0.8rem;color:#64748b;margin:0 0 0.75rem'>
                            Upload a short audio clip to use as a melodic reference.
                            <b>Requires the MusicGen Melody model.</b>
                            </p>""")
                            melody_in = gr.Audio(
                                label="Reference Melody (optional)",
                                type="filepath",
                                sources=["upload"],
                            )

                        gen_btn = gr.Button(
                            "🎵  Generate Music",
                            variant="primary",
                            size="lg",
                        )

                        # Example prompts
                        gr.Examples(
                            examples=[[p] for p in EXAMPLE_PROMPTS[:8]],
                            inputs=[prompt_box],
                            label="💡 Click an example prompt to load it",
                            examples_per_page=4,
                        )

                    # ── Right column — output ──────────────────────────
                    with gr.Column(scale=5, min_width=340):

                        audio_out = gr.Audio(
                            label="🎵 Generated Audio",
                            type="filepath",
                            interactive=False,
                        )
                        vis_out = gr.Image(
                            label="📊 Audio Analysis",
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                        )
                        info_out = gr.Markdown(
                            value=(
                                "_Your generated music and analysis will appear here.\n\n"
                                "Choose a model, write a description, and click **Generate Music**._"
                            ),
                        )

                # Wire up the button
                gen_btn.click(
                    fn=run_generation,
                    inputs=[
                        prompt_box, model_dd, duration_sl,
                        temp_sl, topk_sl, topp_sl, cfg_sl,
                        melody_in,
                    ],
                    outputs=[audio_out, vis_out, info_out],
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — PROMPT GUIDE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("💡  Prompt Guide"):
                gr.Markdown("## How to write great music prompts")

                tips_html = "".join(
                    f'<div class="tip-card">{t}</div>' for t in PROMPT_TIPS
                )
                gr.HTML(f'<div class="tips-grid">{tips_html}</div>')

                gr.Markdown("## Full example prompt library")
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_PROMPTS],
                    inputs=[],
                    label="All example prompts",
                    examples_per_page=6,
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — MODEL GUIDE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🤖  Models"):
                gr.Markdown("""
## MusicGen Model Comparison

| Model | Params | Quality | Speed | Best Use |
|---|---|---|---|---|
| **Small** | 300 M | ★★★☆☆ | ⚡ Fast | Rapid prototyping, demos |
| **Medium** | 1.5 B | ★★★★☆ | 🔄 Medium | Daily use, good balance |
| **Large** | 3.3 B | ★★★★★ | 🐢 Slow | Final outputs, max quality |
| **Melody** | 1.5 B | ★★★★☆ | 🔄 Medium | Melody-guided generation |

## Parameter Reference

### 🌡️ Temperature
Controls sampling randomness.
- **0.5 – 0.8** → conservative, coherent, predictable
- **1.0** → balanced (default)
- **1.2 – 1.5** → experimental, more harmonic surprises
- **1.6+** → highly unpredictable / avant-garde

### 🎯 CFG Coefficient (Classifier-Free Guidance)
How closely the model follows your text prompt.
- **1 – 2** → model has lots of creative freedom
- **3** → balanced (default)
- **5 – 7** → very prompt-adherent
- **8+** → can sound over-constrained / artefact-prone

### 🔝 Top-K
Limits token candidates to the top K at each step.
250 is a solid default. Lower values → more conservative.

### 🎲 Top-P (Nucleus Sampling)
Considers only tokens whose cumulative probability ≥ P.
Set to 0 to use Top-K instead (recommended for music).

---
## Hardware & Deployment

| Setup | Generation Speed (10 s audio) |
|---|---|
| NVIDIA A100 | ~4 s |
| NVIDIA T4 (HuggingFace free) | ~15–25 s |
| Apple M2 Pro (MPS) | ~35 s |
| CPU only | ~3–6 min |

**HuggingFace Spaces** — zero-cost GPU (T4). See `README.md` for deployment steps.
""")

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — ABOUT
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️  About"):
                gr.Markdown(f"""
## AI Music Generation Studio

A production-ready text-to-music application built on top of
[Meta's AudioCraft / MusicGen](https://github.com/facebookresearch/audiocraft).

### Architecture

```
User prompt
    │
    ▼
Gradio UI (app.py)
    │
    ├── MusicGenerator (src/generator.py)
    │       └── MusicGen (facebook/musicgen-*)
    │
    └── AudioUtils (src/audio_utils.py)
            ├── librosa  — analysis
            └── matplotlib — visualisation
```

### Tech Stack

| Component | Library / Tool |
|---|---|
| Music model | `audiocraft` (Meta) |
| UI | `gradio` |
| Audio analysis | `librosa`, `soundfile` |
| Visualisation | `matplotlib` |
| Deployment | HuggingFace Spaces |
| Container | Docker |

### Project Structure

```
music-generation-ai/
├── app.py                  ← entry point
├── src/
│   ├── config.py           ← all settings
│   ├── generator.py        ← MusicGen wrapper
│   └── audio_utils.py      ← visualisation & analysis
├── outputs/                ← generated audio (git-ignored)
├── notebooks/
│   └── MusicGen_Colab.ipynb
├── requirements.txt
├── Dockerfile
└── README.md
```

### License

MIT — free to use, modify, and deploy.
Built for the CodeAlpha AI Internship — Task 3: Music Generation with AI.
""")

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-wrap">
          Built with ❤️ using
          <a href="https://github.com/facebookresearch/audiocraft" target="_blank">Meta AudioCraft</a> ·
          <a href="https://www.gradio.app" target="_blank">Gradio</a> ·
          <a href="https://huggingface.co/facebook/musicgen-small" target="_blank">HuggingFace MusicGen</a>
          <br>CodeAlpha AI Internship — Task 3
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
        favicon_path=None,
    )
