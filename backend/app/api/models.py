import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.model_manager import model_manager, DiffusionArchitectureInfo

logger = logging.getLogger(__name__)
router = APIRouter()

class RemoteWorkerConfigRequest(BaseModel):
    remote_url: str

@router.get("/status")
def get_status():
    return model_manager.get_status()

@router.get("/architecture")
def get_diffusion_architecture():
    """Returns academic & theoretical details: Diffusion vs GAN vs VAE, Forward/Reverse diffusion, U-Net."""
    return DiffusionArchitectureInfo.get_summary()

@router.post("/configure-remote")
def configure_remote_worker(req: RemoteWorkerConfigRequest):
    url = req.remote_url.strip().rstrip("/")
    model_manager.remote_url = url
    # Update active environment
    import os
    os.environ["REMOTE_ML_URL"] = url
    logger.info(f"Updated remote GPU worker URL to: {url}")
    status = model_manager.get_status()
    return {
        "status": "success",
        "remote_url": url,
        "remote_connected": status.get("remote_connected", False),
        "details": status
    }

@router.post("/clear-vram")
def clear_vram():
    model_manager.clear_vram()
    return {"status": "success", "message": "VRAM cache cleared"}
