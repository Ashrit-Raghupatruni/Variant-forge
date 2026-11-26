# VariantForge AI: Multi-Platform Creative Editor & AI Redesign Studio

[![GitHub License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Diffusers](https://img.shields.io/badge/Diffusion-Diffusers%20SDXL-FFD21E.svg?logo=huggingface)](https://huggingface.co/docs/diffusers)
[![Tesla T4 GPU](https://img.shields.io/badge/GPU-Tesla%20T4%20(16GB)-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)

**VariantForge AI** is a production-grade AI/ML creative engineering platform built for digital growth marketers, e-commerce designers, and creative directors. It unifies deep-learning vision-language models, deterministic optical character recognition (OCR), flow-matching generative diffusion, structural spatial conditioning, promptable segmentation, and computer vision algorithms into a seamless, high-performance web studio.

---

## 🌟 Key Capabilities & Deep Learning Architecture

### 1. Dual Execution Modes: Colab GPU Worker & Managed Cloud APIs
VariantForge AI supports flexible, fault-tolerant execution environments:
- **Dedicated Google Colab T4 GPU Worker**: A zero-auth, GPU worker exposing high-throughput neural pipelines over Cloudflare Tunnel with automated CPU memory offloading (`enable_model_cpu_offload()`) to prevent VRAM exhaustion on 16GB cards.
- **Managed Cloud AI APIs**: Direct cloud execution via **Fal.ai** (for sub-3-second SDXL inpainting, ControlNet Canny, and BiRefNet cutouts on H100 GPUs) and **Google Gemini 1.5 Flash** (for vision-language design critiques and multilingual translation).

### 2. SDXL Inpainting (DDPM Reverse Diffusion)
- **Targeted Reverse Diffusion**: Selectively denoises masked areas while preserving untouched image regions with zero pixel drift using `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.
- **Interactive Dual-Layer Canvas**: Draw semi-transparent brush masks directly over the creative underlay with real-time brush size adjustments, mask clearing, and instant high-resolution preview.

### 3. Spatial ControlNet Guidance (Canny Edge Conditioning)
- **Edge-Constrained Diffusion**: Uses `diffusers/controlnet-canny-sdxl-1.0` to preserve product silhouettes, label borders, and typography bounds during creative background generation.
- **Adaptive Resolution Pipeline**: Scales Canny edge maps to 1024×1024 tensor spaces for optimal U-Net latent conditioning with configurable guidance scale and negative prompt filtering.

### 4. AI-Powered Creative Redesign Studio
- **Vision-Language Understanding (Qwen2.5-VL / Gemini 1.5 Flash)**: Scores input designs (1–100) across visual hierarchy, typography balance, color harmony, and lighting depth.
- **Deterministic Text Extraction (EasyOCR / PaddleOCR)**: Automatically detects and preserves brand headlines, subtext, dates, and CTA copy with zero hallucination.
- **3 Distinct Redesign Directions**:
  1. *Modern Minimalist*: Clean, high-key Scandinavian podium layout with soft morning shadows.
  2. *Bold Promotional*: High-contrast commercial studio with dramatic rim highlights.
  3. *Editorial Layout*: Refined typographic framing, cinematic color grading, and magazine-grade presentation.

### 5. Multi-Platform Campaign Matrix Generator
Generates **16 production-ready marketing assets** from a single product upload:
- **Salient Background Removal (RMBG / U2Net / BiRefNet)**: Extracts precise alpha transparencies for bottles, electronics, apparel, and models.
- **Contextual Inpainting & Diffusion (SDXL / FLUX)**: Generates 4 distinct environments (Luxury Studio, Minimalist Podium, Botanical Nature, Premium Dark Commercial).
- **Multi-Platform Aspect Ratio Transformation (OpenCV & Pillow)**:
  - **Instagram Square** (`1080 × 1080` | `1:1`)
  - **Instagram Portrait** (`1080 × 1350` | `4:5`)
  - **Instagram Story / Reel** (`1080 × 1920` | `9:16`)
  - **YouTube Thumbnail** (`1280 × 720` | `16:9`)
  - **LinkedIn Post** (`1200 × 627` | `1.91:1`)
- **Hierarchical ZIP Export**: Bundles all variations and localized assets into structured folders (`VariantForge_Output/Variation_XX/...`).

### 6. Multilingual Localization & Typography Reflow
- Translates brand copy into **English, Hindi (हिंदी), and Telugu (తెలుగు)**.
- Dynamic typography reflow algorithms automatically recalculate font size and bounding boxes based on language character widths to prevent UI clipping.

---

## 🏗️ System Architecture

```
VariantForge-AI/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI server & route mounting
│   │   ├── config.py                        # App configuration & API key management
│   │   ├── api/
│   │   │   ├── images.py                    # Image uploads & binary retrieval
│   │   │   ├── inpainting.py                # SDXL Inpainting (DDPM Reverse Diffusion)
│   │   │   ├── controlnet.py                # ControlNet Canny preview & generation
│   │   │   ├── campaign.py                  # Multi-platform Campaign Matrix & ZIP bundle
│   │   │   ├── redesign.py                  # Vision-Language audit & 3-concept redesign
│   │   │   ├── models.py                    # VRAM diagnostics & remote worker configuration
│   │   │   └── creative.py                  # Aspect ratio conversion & localization
│   │   ├── services/
│   │   │   ├── cloud_ai_service.py          # Unified Fal.ai, Gemini & Together AI client
│   │   │   ├── inpainting_service.py        # SDXL Inpainting pipeline
│   │   │   ├── controlnet_service.py        # ControlNet Canny spatial engine
│   │   │   ├── background_removal_service.py# RMBG-2.0 / U2Net cutout engine
│   │   │   ├── campaign_pipeline_service.py # 7-stage Campaign Matrix pipeline
│   │   │   ├── redesign_pipeline_service.py # 6-stage AI image redesign pipeline
│   │   │   ├── image_understanding_service.py # Gemini & Qwen design auditor
│   │   │   ├── ocr_service.py               # EasyOCR / PaddleOCR text extractor
│   │   │   ├── image_processing_service.py  # OpenCV & Pillow smart resizer
│   │   │   └── model_manager.py             # Singleton GPU cache & VRAM manager
│   │   ├── core/
│   │   │   ├── storage.py                   # Persistent upload and asset manager
│   │   │   └── job_queue.py                 # Asynchronous background job registry
│   │   └── schemas/
│   │       └── api_schemas.py               # Pydantic request & response validation
│   └── tests/                               # Comprehensive unit & integration tests
│
├── frontend/
│   └── public/
│       ├── index.html                       # Unified UI Studio (Canvas, Inpaint, Matrix, Settings)
│       └── app.js                           # Master controller (Fabric.js, toasts, Undo/Redo)
│
├── ml_worker/
│   └── VariantForge_GPU_Inference.ipynb     # 1-Click Colab GPU Worker (Tesla T4 / A100)
│
└── docs/
    ├── MODEL_VERIFICATION_REPORT.md         # Benchmarks & GPU verification report
    ├── DHANUSH_PERFUME_OUTPUTS.md           # Visual verification on Dhanush Perfume
    └── CLOUD_API_MIGRATION_PLAN.md          # Cloud API migration architecture plan
```

---

## ⚡ Quickstart Guide

### 1. Local Backend Setup

```bash
# Clone the repository
git clone https://github.com/Ashrit-Raghupatruni/Variant-forge.git
cd Variant-forge/backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # On Windows
# source venv/bin/activate   # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

- **Web Application**: Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Docs**: Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Setting Up Compute: Choose Your Engine

#### Option A: Free Google Colab T4 GPU Worker (Recommended for Free Tier)
1. Open [`ml_worker/VariantForge_GPU_Inference.ipynb`](ml_worker/VariantForge_GPU_Inference.ipynb) in Google Colab.
2. Select **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**.
3. Run the complete production worker cell.
4. Copy the public Cloudflare tunnel URL displayed in the console:
   ```
   ===========================================================================
   🔥 VARIANTFORGE AI REMOTE T4 GPU WORKER ONLINE!
   👉 COPY THIS PUBLIC URL INTO YOUR LOCAL APP: https://xxxx.trycloudflare.com
   ===========================================================================
   ```
5. In your local app, open **Settings**, paste the URL into **Remote Worker URL**, and click **Connect**.

#### Option B: Managed Cloud AI APIs (Fastest & Zero Colab Maintenance)
1. Obtain an API key from [Fal.ai](https://fal.ai/dashboard/keys) (\$10 free credit) and/or [Google AI Studio](https://aistudio.google.com/app/apikey) (Free Gemini Flash).
2. Open **Settings** in the VariantForge AI dashboard.
3. Paste your keys into the **⚡ Managed Cloud AI (API Keys)** section and click **Save Cloud Keys**.
4. Every generation executes in 2–3 seconds on cloud H100 GPUs with zero session drops.

---

## 📊 End-to-End Model Verification & Benchmarks

All models have been verified on actual production inputs (including real perfume bottles and commercial advertising packaging):

| Task | Neural Architecture | Resolution | Device | Latency | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Foreground Isolation** | U2Net / RMBG-2.0 | `1024 × 1024` | Tesla T4 GPU | ~0.8s | ✅ Verified |
| **Text Extraction** | EasyOCR / CRAFT + CRNN | Native | Tesla T4 GPU | ~1.1s | ✅ Verified |
| **Inpainting** | SDXL Inpainting (DDPM) | `1024 × 1024` | Tesla T4 GPU | ~5.8s | ✅ Verified |
| **Spatial Conditioning** | ControlNet Canny SDXL | `1024 × 1024` | Tesla T4 GPU | ~6.2s | ✅ Verified |
| **Design Understanding** | Gemini 1.5 Flash / Qwen-VL | Multimodal | Cloud API | ~1.4s | ✅ Verified |
| **Campaign Matrix** | End-to-end (16 outputs) | Multi-ratio | Mixed Pipeline| ~24.0s | ✅ Verified |

Detailed visual comparison reports and KaTeX architecture summaries:
- 📖 [Model Verification Report](docs/MODEL_VERIFICATION_REPORT.md)
- 🧴 [Personalized Perfume Case Study](docs/DHANUSH_PERFUME_OUTPUTS.md)

---

## 🔌 API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/images/upload` | Upload source asset and retrieve image metadata |
| `GET` | `/api/images/{image_id}` | Retrieve persistent image file |
| `POST` | `/api/inpaint/` | SDXL Inpainting with DDPM Reverse Diffusion |
| `POST` | `/api/controlnet/canny-preview` | Extract Canny structural edge boundaries |
| `POST` | `/api/controlnet/generate` | Structure-guided diffusion via ControlNet SDXL |
| `POST` | `/api/campaign/matrix` | Generate complete multi-platform campaign matrix |
| `GET` | `/api/campaign/export-zip/{zip_id}` | Download hierarchical campaign archive |
| `POST` | `/api/redesign/redesigns-and-audit` | Vision-Language design audit + 3 concept variations |
| `POST` | `/api/ai/remove-bg` | Background removal and alpha cutout |
| `POST` | `/api/ai/ocr` | Deterministic text extraction with bounding boxes |
| `POST` | `/api/creative/resize` | Aspect ratio resizing (`1:1`, `4:5`, `9:16`, `16:9`) |
| `POST` | `/api/creative/localize` | Translate copy into Hindi/Telugu with typography reflow |
| `GET` | `/api/models/status` | Real-time VRAM telemetry & model lifecycle state |
| `POST` | `/api/models/configure-remote` | Connect remote Google Colab GPU tunnel |
| `POST` | `/api/models/configure-cloud-keys`| Configure Fal.ai / Gemini Cloud API keys |

---

## 📄 License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
