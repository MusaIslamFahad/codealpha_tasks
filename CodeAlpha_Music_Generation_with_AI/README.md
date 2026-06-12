---
title: AI Music Generation Studio
emoji: 🎵
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
short_description: Text-to-music generation powered by Meta's MusicGen
---

# 🎵 AI Music Generation Studio

<div align="center">

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-Backend-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![AudioCraft](https://img.shields.io/badge/Meta-AudioCraft-0064E0?style=for-the-badge)](https://github.com/facebookresearch/audiocraft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship-orange?style=for-the-badge)](https://codealpha.tech)

**Text-to-music generation powered by Meta's MusicGen - describe any sound in plain English and get a studio-quality WAV in seconds.**

> CodeAlpha AI Internship · Task 3: Music Generation with AI

</div>

---

## 📖 Overview

This project is an interactive **AI Music Generation Studio** built as part of the [CodeAlpha](https://www.codealpha.tech/) AI/ML Internship. It wraps Meta's **MusicGen** (from AudioCraft) in a clean Gradio web interface - allowing anyone to generate original music from a text description, optionally guided by a melody audio clip.

The studio runs on **HuggingFace Spaces** (free T4 GPU), **Google Colab**, locally, or inside **Docker** - with zero API costs and no sign-up required to try.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎼 **Text-to-Music** | Describe any style, mood, or instrumentation in plain English and receive a WAV file |
| 🎹 **Melody Conditioning** | Upload an audio clip to guide the melodic direction of the output |
| 🤖 **4 Model Sizes** | Small (fast, low VRAM) → Large (best quality, needs GPU) |
| 📊 **Audio Analysis** | Waveform, mel spectrogram, and chromagram visualisation via librosa |
| ⚙️ **Full Parameter Control** | Temperature, top-k, top-p, and CFG coefficient all adjustable from the UI |
| 🚀 **Zero-cost Deployment** | Runs on HuggingFace Spaces free T4 GPU out of the box |
| 🐳 **Docker Ready** | One command to containerise and run anywhere |
| 📥 **WAV Download** | Generated audio is immediately available to download from the UI |

---

## 🚀 Quick Start

### Option A: HuggingFace Spaces *(Recommended · Free GPU)*

1. Fork this repository on GitHub
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Choose **Gradio** SDK → link your GitHub repo
4. Select **T4 GPU** hardware *(free tier available)*
5. Done - your public URL is live in ~2 minutes

### Option B: Google Colab *(Free GPU)*

1. Open `notebooks/MusicGen_Colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells
4. A public Gradio share link will be printed automatically

### Option C: Local Installation

```bash
# 1. Clone
git clone https://github.com/MusaIslamFahad/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha_Music_Generation_with_AI

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
# Open http://localhost:7860
```

> **GPU strongly recommended.** MusicGen runs on CPU but is very slow. For best results use a CUDA-enabled GPU with at least 6 GB VRAM.

### Option D: Docker

```bash
# Build
docker build -t music-gen-studio .

# Run with GPU (recommended)
docker run -p 7860:7860 --gpus all music-gen-studio

# Run on CPU only
docker run -p 7860:7860 music-gen-studio

# Open http://localhost:7860
```

---

## 🤖 Models

| Model | Parameters | Quality | Speed | VRAM Required |
|---|---|---|---|---|
| MusicGen Small | 300 M | ★★★☆☆ | ⚡ Fast | ~2 GB |
| MusicGen Medium | 1.5 B | ★★★★☆ | 🔄 Medium | ~6 GB |
| MusicGen Large | 3.3 B | ★★★★★ | 🐢 Slow | ~12 GB |
| MusicGen Melody | 1.5 B | ★★★★☆ | 🔄 Medium | ~6 GB |

> **Free Colab / HuggingFace T4 (15 GB VRAM):** comfortably runs Small, Medium, and Melody.  
> **Large** requires a 16 GB+ GPU (A100 or equivalent).

---

## 📁 Project Structure

```
CodeAlpha_Music_Generation_with_AI/
│
├── app.py                        # Gradio application - entry point
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # All settings, model list, example prompts
│   ├── generator.py              # MusicGen wrapper with device management
│   └── audio_utils.py            # Spectrogram / waveform visualisation
│
├── outputs/                      # Generated WAV files (git-ignored)
│
├── examples/
│   └── sample_prompts.txt        # 40+ curated example prompts
│
├── notebooks/
│   └── MusicGen_Colab.ipynb      # Google Colab notebook (free GPU)
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🧠 Architecture

```
User Prompt (text) + optional Melody Audio
            │
            ▼
   ┌─────────────────┐
   │   Gradio UI     │  ← app.py
   │  (web browser)  │    parameter sliders, audio upload,
   │                 │    waveform + spectrogram display
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ MusicGenerator  │  ← src/generator.py
   │  (lazy loader)  │    loads model on first request,
   │                 │    caches on GPU/CPU, handles device mgmt
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────┐
   │   MusicGen (AudioCraft)         │  ← facebook/musicgen-{small,medium,large,melody}
   │   EnCodec tokeniser → LM → DAC  │     via HuggingFace Hub (auto-downloaded)
   └────────┬────────────────────────┘
            │
          WAV file
            │
            ▼
   ┌─────────────────┐
   │   AudioUtils    │  ← src/audio_utils.py
   │   (librosa)     │    waveform plot, mel spectrogram, chromagram
   └─────────────────┘
            │
            ▼
    UI display + download link
```

**Step-by-step:**

1. **Prompt**: The user types a description and adjusts sampling parameters in the Gradio sidebar.
2. **Load**: `MusicGenerator` lazily loads the chosen MusicGen variant from HuggingFace Hub on the first call, then caches it in memory for subsequent generations.
3. **Generate**: MusicGen encodes the text with a T5 encoder, autoregressively samples audio tokens with the language model, and decodes them to a waveform via EnCodec.
4. **Analyse**: `AudioUtils` uses librosa to render a waveform plot, mel spectrogram, and chromagram for the generated clip.
5. **Output**: The WAV file and visualisation plots are returned to the Gradio UI, where the user can play, inspect, and download.

---

## ⚙️ Parameter Reference

| Parameter | Range | Default | Effect |
|---|---|---|---|
| **Duration** | 1 – 30 s | 10 s | Length of the generated clip |
| **Temperature** | 0.1 – 2.0 | 1.0 | Randomness - higher values = more creative and varied output |
| **CFG Coefficient** | 1 – 10 | 3.0 | Prompt adherence - higher values = output stays closer to the text |
| **Top-K** | 0 – 1000 | 250 | Restricts sampling to the top-K most likely tokens at each step |
| **Top-P** | 0.0 – 1.0 | 0.0 | Nucleus sampling threshold (set to 0 to use Top-K instead) |

---

## 💡 Prompt Writing Guide

Great prompts produce great music. Here are the key rules:

### ✅ Be specific about instruments

```
✅ "Jazz piano trio with upright bass and brushed snare drum"
❌ "Jazz music"
```

### ✅ Include mood, setting, and era

```
✅ "Melancholic solo violin with slow vibrato, echoing in an empty cathedral, late-night atmosphere"
❌ "Sad violin"
```

### ✅ Reference tempo and genre precisely

```
✅ "80s synth-pop at 120 BPM, arpeggiated lead synth, gated reverb drums, driving bass line"
❌ "Electronic music"
```

### ✅ Layer descriptors for richer output

```
✅ "Cinematic orchestral swell, French horns and strings, building tension, Hans Zimmer style"
✅ "Lo-fi hip-hop beat, mellow Rhodes piano, rain ambience, 70 BPM, vinyl crackle"
```

### 📝 Sample Prompts

| Genre | Example Prompt |
|---|---|
| Cinematic | `Orchestral film score, sweeping strings, dramatic French horns, building climax` |
| Lo-fi | `Lo-fi hip hop, mellow piano, rain sounds, soft drums, 75 BPM` |
| Electronic | `Ambient techno, hypnotic 4/4 kick, shimming hi-hats, evolving synth pads, 128 BPM` |
| Classical | `Solo classical guitar, fingerpicking arpeggios, Spanish flamenco influence` |
| Jazz | `Upbeat swing jazz, trumpet lead, walking bass, brushed snare, piano comping` |
| Acoustic | `Acoustic folk duo, fingerpicked guitar and gentle fiddle, storytelling ballad` |

---

## 📋 Requirements

```
# Core
torch>=2.0.0
audiocraft>=1.0.0      # Meta's MusicGen
gradio>=4.26.0

# Audio analysis & visualisation
librosa>=0.10.0
matplotlib>=3.7.0
soundfile>=0.12.0
numpy>=1.24.0

# Utilities
scipy>=1.11.0
```

Python version: **3.10 or higher**

> Install everything in one step: `pip install -r requirements.txt`

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

**Ideas for contributions:** additional model support, batch generation, MIDI export, custom training integration, or UI themes.

---

## 👤 Author

**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University, Dhaka  
📧 musa.islam.fahad@gmail.com  
🌐 [Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.  
Free to use, modify, and deploy.

---

## 🙏 Acknowledgements

- [Meta AudioCraft](https://github.com/facebookresearch/audiocraft) - MusicGen model and EnCodec
- [HuggingFace](https://huggingface.co) - Model hosting and free GPU Spaces
- [Gradio](https://gradio.app) - Web UI framework
- [librosa](https://librosa.org) - Audio analysis and visualisation
- [CodeAlpha](https://www.codealpha.tech/) - Internship opportunity and project brief

---

<div align="center">

Made with 🎵 as part of the **CodeAlpha AI/ML Internship**

**[⬆ Back to Top](#-ai-music-generation-studio)**

</div>
