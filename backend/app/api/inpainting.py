import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.storage import storage_manager
from app.services.inpainting_service import InpaintingService
from app.models.entities import VariantRepository

logger = logging.getLogger(__name__)
router = APIRouter()

class InpaintRequest(BaseModel):
    project_id: Optional[str] = None
    image_id: str
    mask_image_id: str
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, artifacts"
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    strength: Optional[float] = 0.95
    seed: Optional[int] = 42

@router.post("/")
def run_inpaint(req: InpaintRequest):
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("SDXL Inpainting (DDPM)")

    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Source image not found")
    
    mask = storage_manager.get_mask(req.mask_image_id) or storage_manager.get_image(req.mask_image_id)
    if not mask:
        raise HTTPException(status_code=404, detail="Mask image not found")

    out_img = InpaintingService.inpaint(
        image=img,
        mask=mask,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        strength=req.strength,
        seed=req.seed
    )

    res_id = storage_manager.save_image(out_img, category="inpainting")
    res_url = f"/api/images/{res_id}"

    if req.project_id:
        VariantRepository.create({
            "project_id": req.project_id,
            "platform": "custom_inpaint",
            "width": out_img.width,
            "height": out_img.height,
            "design_style": "AI Inpainting",
            "generation_model": "SDXL Inpaint (DDPM)",
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
