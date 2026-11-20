from fastapi import APIRouter, HTTPException
from app.core.job_queue import job_registry
from app.schemas.api_schemas import JobResponse

router = APIRouter()

@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Retrieves asynchronous model inference task status."""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JobResponse(**job)
