import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.entities import ProjectRepository, AnalysisRepository, VariantRepository
from app.core.storage import storage_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class ProjectCreateRequest(BaseModel):
    name: str
    campaign_name: Optional[str] = ""
    category: Optional[str] = "general"
    brand_name: Optional[str] = ""
    description: Optional[str] = ""
    target_platforms: Optional[List[str]] = ["instagram_square", "instagram_portrait", "instagram_story", "youtube_thumbnail"]
    target_regions: Optional[List[str]] = ["Global"]
    preferred_language: Optional[str] = "en"
    original_image_id: Optional[str] = None

class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    campaign_name: Optional[str] = None
    category: Optional[str] = None
    brand_name: Optional[str] = None
    description: Optional[str] = None
    target_platforms: Optional[List[str]] = None
    target_regions: Optional[List[str]] = None
    preferred_language: Optional[str] = None
    original_image_id: Optional[str] = None
    original_image_url: Optional[str] = None

@router.get("/")
def list_projects():
    return {"projects": ProjectRepository.list_all()}

@router.post("/")
def create_project(req: ProjectCreateRequest):
    data = req.dict()
    if req.original_image_id:
        data["original_image_url"] = f"/api/images/{req.original_image_id}"
    project = ProjectRepository.create(data)
    return project

@router.get("/{project_id}")
def get_project(project_id: str):
    p = ProjectRepository.get(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    # Attach analysis and variants
    analysis = AnalysisRepository.get_by_project(project_id)
    variants = VariantRepository.list_by_project(project_id)
    return {
        "project": p,
        "analysis": analysis,
        "variants": variants
    }

@router.put("/{project_id}")
def update_project(project_id: str, req: ProjectUpdateRequest):
    updates = {k: v for k, v in req.dict().items() if v is not None}
    updated = ProjectRepository.update(project_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return updated

@router.delete("/{project_id}")
def delete_project(project_id: str):
    success = ProjectRepository.delete(project_id)
    return {"status": "success", "deleted": success}
