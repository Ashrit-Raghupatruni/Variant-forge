import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.core.storage import storage_manager
from app.core.job_queue import job_registry, JobStatus
from app.schemas.api_schemas import JobResponse
from app.services.redesign_pipeline_service import RedesignPipelineService

logger = logging.getLogger(__name__)
router = APIRouter()

class RedesignRequest(BaseModel):
    image_id: str
    user_instructions: Optional[str] = "Redesign this image with a modern and professional appearance. Improve typography, spacing, lighting and layout while retaining all original information."

def run_redesign_task(job_id: str, image_id: str, instructions: str):
    try:
        def on_progress(pct: int, msg: str):
            job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=pct)

        result = RedesignPipelineService.execute_redesign(
            image_id=image_id,
            user_instructions=instructions,
            progress_callback=on_progress
        )

        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100
        )
        job_record = job_registry.get_job(job_id)
        if job_record:
            job_record["metadata"] = result

    except Exception as e:
        logger.error(f"Redesign task {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

@router.post("/execute", response_model=JobResponse)
@router.post("/redesigns-and-audit")
async def execute_image_redesign(req: RedesignRequest, background_tasks: BackgroundTasks):
    """
    Triggers the AI Image Redesign Pipeline (Qwen2.5-VL + PaddleOCR + FLUX.2 Klein + Layer Reconstruction).
    Supports synchronous return or background job.
    """
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("AI Redesign Studio (Qwen2.5-VL & FLUX.2)")

    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        result = RedesignPipelineService.execute_redesign(
            image_id=req.image_id,
            user_instructions=req.user_instructions
        )
        return {
            "job_id": "direct_redesign",
            "task_type": "AI_IMAGE_REDESIGN",
            "status": "COMPLETED",
            "progress": 100,
            "metadata": result
        }
    except Exception as e:
        job_id = job_registry.create_job("AI_IMAGE_REDESIGN")
        background_tasks.add_task(run_redesign_task, job_id, req.image_id, req.user_instructions)
        return job_registry.get_job(job_id)
