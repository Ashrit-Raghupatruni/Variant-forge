import os
import io
import zipfile
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.entities import VariantRepository, ProjectRepository
from app.core.storage import storage_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class ExportZipRequest(BaseModel):
    project_id: str
    variant_ids: Optional[List[str]] = None

@router.post("/zip")
def export_campaign_zip(req: ExportZipRequest):
    project = ProjectRepository.get(req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    all_variants = VariantRepository.list_by_project(req.project_id)
    if req.variant_ids:
        selected_variants = [v for v in all_variants if v["id"] in req.variant_ids]
    else:
        selected_variants = all_variants

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, var in enumerate(selected_variants):
            img_id = var.get("image_id")
            if not img_id:
                continue
            img = storage_manager.get_image(img_id)
            if not img:
                continue
            
            platform = var.get("platform", "variant")
            lang = var.get("language", "en")
            filename = f"{project['name'].replace(' ', '_')}_{platform}_{lang}_{idx+1}.png"
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            zip_file.writestr(filename, img_byte_arr.getvalue())

    zip_buffer.seek(0)
    campaign_clean = project["name"].replace(" ", "_")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=VariantForge_{campaign_clean}_Campaign.zip"}
    )
