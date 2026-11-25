# VariantForge AI: Multi-Platform Creative Editor & AI Redesign System

VariantForge AI is a production-oriented AI/ML creative engineering platform built for growth marketing teams. It combines deep-learning vision-language models, deterministic optical character recognition (OCR), flow-matching generative diffusion, promptable segmentation, and computer vision algorithms into a unified creative suite.

---

## Key Capabilities & Pre-Trained Deep Learning Integrations

### 1. AI-Powered Image Redesign System
Transforms existing posters, flyers, banners, and advertisements into modern, professionally redesigned assets while preserving all original text and information:
- **Vision-Language Understanding (Qwen2.5-VL)**: Analyzes design quality, contrast, balance, lighting depth, and visual hierarchy flaws.
- **Deterministic Text Extraction (PaddleOCR)**: Extracts headlines, subheadings, dates, and call-to-actions with zero hallucination.
- **3 Distinct Redesign Directions**:
  1. *Modern Minimalist*: Clean, spacious editorial layout with Scandinavian natural shadows.
  2. *Premium Professional*: Luxury dark obsidian commercial studio with cinematic rim lighting.
  3. *Creative & Vibrant*: Bold color gradients, dynamic spherical lighting, and energetic composition.
- **Before-and-After Comparison**: Real-time side-by-side inspection between the original uploaded creative and the redesigned outputs.

### 2. Multi-Platform Campaign Matrix Generator
Transforms a single product photograph into **16 production-ready marketing creatives**:
- **Background Removal (BRIA RMBG-2.0)**: High-definition alpha matting for products, packaging, and models.
- **Contextual Inpainting & Generation (FLUX.2 Klein 4B)**: Low-step (4 inference steps) distilled flow-matching generation via Hugging Face `diffusers` (`torch.bfloat16`).
- **4 Stylized Variations**: Luxury Studio, Minimalist White Podium, Botanical Nature, Premium Dark Commercial.
- **Multi-Platform Optimization (OpenCV & Pillow)**:
  - **Instagram Square** (`1080 × 1080` - 1:1)
  - **Instagram Portrait** (`1080 × 1350` - 4:5)
  - **Instagram Story / Reel** (`1080 × 1920` - 9:16)
  - **YouTube Thumbnail** (`1280 × 720` - 16:9)
  - **LinkedIn Post** (`1200 × 627` - 1.91:1)
- **Hierarchical ZIP Export**: Automatically archives into `VariantForge_Output/Variation_XX/...` structure.

### 3. Interactive Object Segmentation (Meta SAM 2)
- Point-and-click promptable segmentation (`SAM2ImagePredictor`).
- Extracts clicked products and items into independent, draggable, resizable Fabric.js canvas layers.

### 4. Multilingual Localization Engine
- Translates marketing copy into **English, Hindi (हिंदी), and Telugu (తెలుగు)**.
- Dynamic typography reflow algorithms adjust font bounds and sizing to prevent text overflow across languages.

### 5. SDXL Inpainting (DDPM Reverse Diffusion) & ControlNet
- **Targeted Reverse Diffusion**: Denoises masked regions while strictly preserving unmasked creative content using `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.
- **Structural ControlNet Guidance (Canny)**: Extracts structural boundaries and edge maps to constrain image generation to exact product packaging, label lines, and physical geometry.
- **Interactive Dual-Layer Canvas**: Real-time mask drawing directly over the creative underlay with live brush sizing, clear controls, and high-resolution preview.
- **Undo / Redo & Layer Deletion**: Full history state stack (`Ctrl+Z`, `Ctrl+Y`) and element layer deletion in the Fabric.js canvas editor.

### 6. Verified Deep Learning Benchmarks & GPU Inference
- For full benchmark logs, image comparisons, and before/after diffusion analysis executed on **Tesla T4 GPU**, see:
  - [Model Verification & Benchmark Report](docs/MODEL_VERIFICATION_REPORT.md)
  - [Personalized Perfume Output Report](docs/DHANUSH_PERFUME_OUTPUTS.md)

---

## Project Architecture

```plaintext
VariantForge-AI/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI application & route mounting
│   │   ├── config.py                        # App configuration & ML provider settings
│   │   ├── api/
│   │   │   ├── images.py                    # POST /api/images/upload, GET /api/images/{image_id}
│   │   │   ├── ai.py                        # POST /api/ai/generate, /edit, /segment, /remove-background
│   │   │   ├── creative.py                  # POST /api/creative/resize, /localize
│   │   │   ├── campaign.py                  # POST /api/campaign/generate-matrix, /export-zip
│   │   │   ├── redesign.py                  # POST /api/redesign/execute
│   │   │   └── jobs.py                      # GET /api/jobs/{job_id} (Asynchronous progress tracker)
│   │   ├── services/
│   │   │   ├── model_manager.py             # Singleton model cache & GPU memory management
│   │   │   ├── ocr_service.py               # PaddleOCR deterministic text extraction
│   │   │   ├── image_understanding_service.py # Qwen2.5-VL vision-language design auditor
│   │   │   ├── redesign_pipeline_service.py # 6-stage image redesign pipeline
│   │   │   ├── campaign_pipeline_service.py # 7-stage campaign matrix & ZIP packager
│   │   │   ├── image_generation_service.py  # FLUX.2 Klein 4B text-to-image
│   │   │   ├── image_editing_service.py     # FLUX.2 Klein 4B image-to-image
│   │   │   ├── segmentation_service.py      # Meta SAM 2 promptable segmentation
│   │   │   ├── background_removal_service.py# BRIA RMBG-2.0 alpha isolation
│   │   │   ├── image_processing_service.py  # OpenCV & Pillow smart resizer
│   │   │   └── localization_service.py      # English, Hindi, Telugu translation & font reflow
│   │   ├── core/
│   │   │   ├── storage.py                   # Persistent upload and asset manager
│   │   │   └── job_queue.py                 # Asynchronous background task registry
│   │   └── schemas/
│   │       └── api_schemas.py               # Pydantic request & response validation
│   ├── tests/
│   │   ├── test_campaign_pipeline.py        # 16-asset matrix & ZIP archive tests
│   │   ├── test_redesign_pipeline.py        # OCR, Qwen visual audit & redesign tests
│   │   └── test_core_services.py            # Resizer, localization & storage tests
│   └── requirements.txt
│
├── frontend/
│   └── public/
│       └── index.html                       # Unified Studio: AI Redesign + Campaign Matrix + Fabric.js Canvas
│
├── ml_worker/
│   ├── worker_app.py                        # Standalone GPU FastAPI server (Colab / Cloud GPU)
│   ├── VariantForge_GPU_Inference.ipynb     # 1-click Google Colab GPU notebook with ngrok tunnel
│   └── requirements_gpu.txt                 # PyTorch CUDA, diffusers, transformers, accelerate
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quickstart

### 1. Run Backend Locally
```powershell
cd backend
.\venv\Scripts\activate
# If on macOS/Linux: source venv/bin/activate

uvicorn app.main:app --reload --port 8000
```
- Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser to access the full VariantForge UI.
- Interactive API documentation available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### 2. Run Test Suite
```powershell
$env:PYTHONPATH = "C:\Users\ashri\GENAI\backend"
pytest backend/tests/
```
All 6 test suites verify:
- OCR text extraction & Qwen2.5-VL design defect assessment
- FLUX.2 Klein 3-direction visual redesign
- 16-asset multi-platform matrix generation & ZIP archiving
- OpenCV/Pillow aspect ratio transformations
- Hindi & Telugu translation and typography reflow bounds

### 3. Dedicated Cloud GPU Worker (Google Colab / Cloud GPU)
Heavy neural inference (SDXL Inpainting, ControlNet, FLUX.2, SAM 2, RMBG) runs seamlessly on a remote Tesla T4 / A100 GPU:
1. Open [`ml_worker/VariantForge_GPU_Inference.ipynb`](ml_worker/VariantForge_GPU_Inference.ipynb) in Google Colab with a GPU runtime (T4, A100, or V100).
2. Run all cells to start the FastAPI worker and expose a public Cloudflare tunnel (`trycloudflare.com`) or ngrok URL.
3. In the UI under **Models & DDPM** (or in `backend/app/config.py`), paste the public worker URL:
   ```python
   REMOTE_ML_URL = "https://your-tunnel.trycloudflare.com"
   ```
4. All heavy diffusion, inpainting, and segmentation inference jobs execute automatically on the cloud GPU and stream results directly back into your local VariantForge studio.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/images/upload` | Upload and validate source images (PNG, JPG, WebP) |
| `GET` | `/api/images/{image_id}` | Retrieve persistent image file |
| `POST` | `/api/inpaint/` | SDXL Inpainting (DDPM Reverse Diffusion on GPU) |
| `POST` | `/api/controlnet/canny-preview` | Extract structural Canny edge map |
| `POST` | `/api/controlnet/generate` | Structure-guided diffusion via ControlNet SDXL |
| `POST` | `/api/redesign/execute` | 6-stage AI image redesign pipeline (Qwen2.5-VL + PaddleOCR + FLUX.2 Klein) |
| `POST` | `/api/campaign/generate-matrix` | Generate 4 variations × 4 platforms = 16 creative assets |
| `GET` | `/api/campaign/export-zip/{zip_id}`| Download hierarchical campaign ZIP bundle |
| `POST` | `/api/ai/remove-background` | RMBG-2.0 / U2Net alpha foreground extraction |
| `POST` | `/api/ai/segment` | Meta SAM 2 point/box promptable segmentation |
| `POST` | `/api/ai/replace-background` | RMBG foreground + FLUX.2 background + contact shadow composite |
| `POST` | `/api/creative/resize` | Exact platform resizing (1080x1080, 1080x1350, 1080x1920, 1280x720) |
| `POST` | `/api/creative/localize` | Marketing copy translation & typography reflow (English, Hindi, Telugu) |
| `POST` | `/api/models/configure-remote` | Dynamically connect remote Google Colab GPU tunnel |
| `GET` | `/api/models/status` | Live VRAM and model readiness status |
| `GET` | `/api/jobs/{job_id}` | Asynchronous job status and progress tracking |
| `GET` | `/health` | Server and compute device status |
