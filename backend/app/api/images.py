import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io

from app.core.storage import storage_manager
from app.schemas.api_schemas import ImageUploadResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Accepts image uploads, validates format, stores to disk, returns image metadata."""
    filename = file.filename or "upload.png"
    ext = filename.split(".")[-1].lower() if "." in filename else "png"
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Allowed: {ALLOWED_EXTENSIONS}")

    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content))
        w, h = pil_img.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Corrupt image file: {str(e)}")

    image_id = storage_manager.save_bytes(content, ext=ext, category="images")

    return ImageUploadResponse(
        image_id=image_id,
        image_url=f"/api/images/{image_id}",
        width=w,
        height=h,
        format=ext.upper()
    )

@router.get("/{image_id}")
async def get_image(image_id: str):
    """Retrieves stored image by image_id."""
    path = storage_manager.get_image_path(image_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = "image/png" if path.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media_type)
