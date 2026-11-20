import os
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.core.storage import storage_manager
from app.core.job_queue import job_registry, JobStatus
from app.schemas.api_schemas import CampaignMatrixRequest, JobResponse
from app.services.campaign_pipeline_service import CampaignPipelineService

logger = logging.getLogger(__name__)
router = APIRouter()

def run_campaign_matrix_task(job_id: str, req_data: dict):
    try:
        def update_progress(pct: int, msg: str):
            job_registry.update_job(job_id, status=JobStatus.PROCESSING, progress=pct)

        result_data = CampaignPipelineService.execute_campaign(
            image_id=req_data["image_id"],
            user_prompt=req_data.get("user_prompt", ""),
            platforms=req_data.get("platforms", ["instagram_square", "instagram_portrait", "instagram_story", "youtube_thumbnail"]),
            num_variations=req_data.get("num_variations", 4),
            target_lang=req_data.get("target_lang", "en"),
            headline=req_data.get("headline"),
            cta=req_data.get("cta"),
            progress_callback=update_progress
        )

        job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_image_url=result_data["zip_url"]
        )
        # Store full matrix data in job record
        job_record = job_registry.get_job(job_id)
        if job_record:
            job_record["metadata"] = result_data

    except Exception as e:
        logger.error(f"Campaign Matrix Job {job_id} failed: {e}")
        job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))

@router.post("/generate-matrix", response_model=JobResponse)
@router.post("/matrix")
async def generate_campaign_matrix(req: CampaignMatrixRequest, background_tasks: BackgroundTasks):
    """
    Launches generation of the complete 4 Variations x Multi-Platform Creative Matrix.
    Executes synchronously or asynchronously and returns the full variant list.
    """
    from app.services.model_manager import model_manager
    model_manager.require_active_gpu("Campaign Matrix Generator (FLUX.2 & RMBG-2.0)")
    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail=f"Image {req.image_id} not found")

    try:
        result_data = CampaignPipelineService.execute_campaign(
            image_id=req.image_id,
            user_prompt=req.user_prompt or "",
            platforms=req.platforms or ["instagram_square", "instagram_portrait", "instagram_story", "youtube_thumbnail"],
            num_variations=req.num_variations or 4,
            target_lang=req.target_lang or "en",
            headline=req.headline,
            cta=req.cta
        )

        # Flatten variants for direct UI rendering
        variants_list = []
        for style_key, s_data in result_data.get("variations", {}).items():
            for p_key, p_data in s_data.get("platforms", {}).items():
                variants_list.append({
                    "style": s_data.get("style_title", style_key),
                    "platform": p_key,
                    "width": p_data["dimensions"][0],
                    "height": p_data["dimensions"][1],
                    "image_id": p_data["image_id"],
                    "image_url": p_data["image_url"]
                })

        return {
            "status": "COMPLETED",
            "progress": 100,
            "zip_url": result_data["zip_url"],
            "variants": variants_list,
            "metadata": result_data
        }
    except Exception as e:
        logger.error(f"Campaign Matrix execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-zip/{zip_id}")
async def export_campaign_zip(zip_id: str):
    """Streams the hierarchical campaign ZIP bundle."""
    path = storage_manager.get_image_path(zip_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Campaign ZIP archive not found")
    return FileResponse(
        path,
        media_type="application/zip",
        filename="VariantForge_Campaign_Creatives.zip"
    )
