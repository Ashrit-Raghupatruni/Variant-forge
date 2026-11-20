import io
import base64
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.storage import storage_manager
from app.services.controlnet_service import ControlNetService
from app.models.entities import VariantRepository

logger = logging.getLogger(__name__)
router = APIRouter()

class CannyPreviewRequest(BaseModel):
    image_id: str
    low_threshold: Optional[int] = 100
    high_threshold: Optional[int] = 200

class ControlNetGenerateRequest(BaseModel):
    project_id: Optional[str] = None
    image_id: str
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, artifacts"
    conditioning_scale: Optional[float] = 0.8
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = 42

@router.post("/canny-preview")
def preview_canny(req: CannyPreviewRequest):
    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Source image not found")
    
    canny_img = ControlNetService.extract_canny(img, req.low_threshold, req.high_threshold)
    canny_id = storage_manager.save_image(canny_img, category="controlnet")
    return {
        "status": "success",
        "canny_image_id": canny_id,
        "canny_image_url": f"/api/images/{canny_id}",
        "width": canny_img.width,
        "height": canny_img.height
    }

@router.post("/generate")
def run_controlnet(req: ControlNetGenerateRequest):
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("ControlNet SDXL")

    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Source image not found")

    out_img = ControlNetService.generate_controlled(
        image=img,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        conditioning_scale=req.conditioning_scale,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed
    )

    res_id = storage_manager.save_image(out_img, category="controlnet")
    res_url = f"/api/images/{res_id}"

    if req.project_id:
        VariantRepository.create({
            "project_id": req.project_id,
            "platform": "controlnet_guided",
            "width": out_img.width,
            "height": out_img.height,
            "design_style": "Structure Preserved",
            "generation_model": "ControlNet SDXL",
            "image_id": res_id,
            "image_url": res_url
        })

    return {
        "status": "success",
        "result_image_id": res_id,
        "result_image_url": res_url,
        "width": out_img.width,
        "height": out_img.height
    }
