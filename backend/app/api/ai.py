import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from PIL import Image

from app.core.storage import storage_manager
from app.core.job_queue import job_registry, JobStatus
from app.schemas.api_schemas import (
    AIGenerateRequest,
    AIEditRequest,
    AISegmentRequest,
    AIRemoveBackgroundRequest,
    AIReplaceBackgroundRequest,
    JobResponse
)
from app.services.image_generation_service import ImageGenerationService
from app.services.image_editing_service import ImageEditingService
from app.services.segmentation_service import SegmentationService
from app.services.background_removal_service import BackgroundRemovalService
from app.services.image_processing_service import ImageProcessingService

logger = logging.getLogger(__name__)
router = APIRouter()

def run_generate_task(job_id: str, prompt: str, w: int, h: int, steps: int, ref_img_id: str = None):
    try:
        job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=20)
        if ref_img_id:
            ref_img = storage_manager.get_image(ref_img_id)
            out_img = ImageEditingService.edit(image=ref_img, instruction_prompt=prompt, num_inference_steps=steps)
        else:
            out_img = ImageGenerationService.generate(prompt=prompt, width=w, height=h, num_inference_steps=steps)
            
        job_registry.update_job(job_id, progress=80)
        res_id = storage_manager.save_image(out_img, category="generations")
        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_image_id=res_id,
            result_image_url=f"/api/images/{res_id}"
        )
    except Exception as e:
        logger.error(f"Generate job {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

def run_edit_task(job_id: str, image_id: str, prompt: str, steps: int):
    try:
        job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=25)
        img = storage_manager.get_image(image_id)
        if not img:
            raise ValueError(f"Image ID {image_id} not found")
        out_img = ImageEditingService.edit(image=img, instruction_prompt=prompt, num_inference_steps=steps)
        res_id = storage_manager.save_image(out_img, category="generations")
        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_image_id=res_id,
            result_image_url=f"/api/images/{res_id}"
        )
    except Exception as e:
        logger.error(f"Edit job {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

def run_remove_bg_task(job_id: str, image_id: str):
    try:
        job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=30)
        img = storage_manager.get_image(image_id)
        if not img:
            raise ValueError(f"Image ID {image_id} not found")
        cutout = BackgroundRemovalService.remove_background(img)
        res_id = storage_manager.save_image(cutout, category="generations", ext="png")
        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_image_id=res_id,
            result_image_url=f"/api/images/{res_id}"
        )
    except Exception as e:
        logger.error(f"Remove BG job {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

def run_replace_bg_task(job_id: str, fg_id: str, bg_prompt: str, steps: int):
    try:
        job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=20)
        fg_img = storage_manager.get_image(fg_id)
        if not fg_img:
            raise ValueError(f"Foreground image {fg_id} not found")
        
        # 1. Ensure foreground is isolated
        if fg_img.mode != "RGBA":
            fg_isolated = BackgroundRemovalService.remove_background(fg_img)
        else:
            fg_isolated = fg_img
            
        job_registry.update_job(job_id, progress=50)
        # 2. Generate new contextual background with FLUX.2 Klein
        bg_gen = ImageGenerationService.generate(prompt=bg_prompt, width=1024, height=1024, num_inference_steps=steps)
        
        job_registry.update_job(job_id, progress=80)
        # 3. Composite product over background
        composite = ImageProcessingService.replace_background(fg_isolated, bg_gen)
        res_id = storage_manager.save_image(composite, category="generations", ext="png")
        
        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_image_id=res_id,
            result_image_url=f"/api/images/{res_id}"
        )
    except Exception as e:
        logger.error(f"Replace BG job {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

@router.post("/generate", response_model=JobResponse)
async def generate_image(req: AIGenerateRequest, background_tasks: BackgroundTasks):
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("FLUX.2 Klein 4B")

    job_id = job_registry.create_job("FLUX_GENERATE")
    background_tasks.add_task(
        run_generate_task, job_id, req.prompt, req.width, req.height, req.num_inference_steps, req.reference_image_id
    )
    return job_registry.get_job(job_id)

@router.post("/edit", response_model=JobResponse)
async def edit_image(req: AIEditRequest, background_tasks: BackgroundTasks):
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("FLUX.2 Klein 4B")

    job_id = job_registry.create_job("FLUX_EDIT")
    background_tasks.add_task(run_edit_task, job_id, req.image_id, req.instruction_prompt, req.num_inference_steps)
    return job_registry.get_job(job_id)

@router.post("/remove-background", response_model=JobResponse)
@router.post("/remove-bg")
async def remove_background(req: AIRemoveBackgroundRequest, background_tasks: BackgroundTasks):
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("BRIA RMBG-2.0")

    # Support immediate synchronous execution if requested
    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        fg_img = BackgroundRemovalService.remove_background(img)
        res_id = storage_manager.save_image(fg_img, category="generations", ext="png")
        return {
            "job_id": "direct_rmbg",
            "task_type": "RMBG_REMOVE",
            "status": "COMPLETED",
            "progress": 100,
            "result_image_id": res_id,
            "result_image_url": f"/api/images/{res_id}"
        }
    except Exception as e:
        job_id = job_registry.create_job("RMBG_REMOVE")
        background_tasks.add_task(run_remove_bg_task, job_id, req.image_id)
        return job_registry.get_job(job_id)

@router.post("/replace-background", response_model=JobResponse)
async def replace_background(req: AIReplaceBackgroundRequest, background_tasks: BackgroundTasks):
    job_id = job_registry.create_job("REPLACE_BACKGROUND")
    background_tasks.add_task(run_replace_bg_task, job_id, req.foreground_image_id, req.background_prompt, req.num_inference_steps)
    return job_registry.get_job(job_id)

@router.post("/segment")
async def segment_object(req: AISegmentRequest):
    """Interactive SAM 2 synchronous promptable segmentation."""
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("Meta SAM 2.1")

    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
        
    cutout, mask = SegmentationService.segment(image=img, points=req.points, labels=req.labels, box=req.box)
    
    cutout_id = storage_manager.save_image(cutout, category="generations", ext="png")
    mask_id = storage_manager.save_image(mask, category="masks", ext="png")
    
    return {
        "status": "COMPLETED",
        "cutout_image_id": cutout_id,
        "cutout_url": f"/api/images/{cutout_id}",
        "mask_image_id": mask_id,
        "mask_url": f"/api/images/{mask_id}"
    }
