import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.images import router as images_router
from app.api.ai import router as ai_router
from app.api.creative import router as creative_router
from app.api.jobs import router as jobs_router
from app.api.campaign import router as campaign_router
from app.api.redesign import router as redesign_router
from app.api.projects import router as projects_router
from app.api.inpainting import router as inpaint_router
from app.api.controlnet import router as controlnet_router
from app.api.models import router as models_router
from app.api.exports import router as exports_router

app = FastAPI(
    title="VariantForge AI: Multi-Platform Creative Editor & Redesign Studio",
    description="Production-grade Creative Adaptation & Redesign Platform combining Diffusion & DDPM, SDXL Inpainting, ControlNet, FLUX.2 Klein, Qwen2.5-VL, RMBG, and SAM 2",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Endpoints
app.include_router(projects_router, prefix="/api/projects", tags=["Projects"])
app.include_router(images_router, prefix="/api/images", tags=["Images"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI Inference"])
app.include_router(inpaint_router, prefix="/api/inpaint", tags=["SDXL Inpainting"])
app.include_router(controlnet_router, prefix="/api/controlnet", tags=["ControlNet Canny"])
app.include_router(creative_router, prefix="/api/creative", tags=["Creative Operations"])
app.include_router(campaign_router, prefix="/api/campaign", tags=["Campaign Pipeline"])
app.include_router(redesign_router, prefix="/api/redesign", tags=["AI Redesign"])
app.include_router(models_router, prefix="/api/models", tags=["Model Architecture & Status"])
app.include_router(exports_router, prefix="/api/exports", tags=["Exports"])
app.include_router(jobs_router, prefix="/api/jobs", tags=["Async Jobs"])

FRONTEND_PUBLIC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "public")
)
FRONTEND_INDEX = os.path.join(FRONTEND_PUBLIC, "index.html")

if os.path.exists(FRONTEND_PUBLIC):
    app.mount("/static", StaticFiles(directory=FRONTEND_PUBLIC), name="static")

@app.get("/")
def serve_ui():
    """Serves the VariantForge AI Workspace UI directly."""
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {
        "message": "VariantForge AI Backend is running!",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    from app.services.model_manager import model_manager
    return {
        "status": "healthy",
        "service": "VariantForge AI Backend",
        "version": "2.5.0",
        "device": model_manager.device,
        "remote_ml_enabled": model_manager.is_remote_enabled(),
        "remote_worker_url": model_manager.remote_url or None
    }
