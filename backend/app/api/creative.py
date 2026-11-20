from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.core.storage import storage_manager
from app.schemas.api_schemas import CreativeResizeRequest, CreativeLocalizeRequest
from app.services.image_processing_service import ImageProcessingService, PLATFORM_SPECS
from app.services.localization_service import LocalizationService

router = APIRouter()

@router.post("/resize")
async def resize_creative(req: CreativeResizeRequest):
    """Generates exact multi-platform creative dimensions for Instagram, YouTube, and LinkedIn."""
    img = storage_manager.get_image(req.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
        
    platforms = req.platforms or list(PLATFORM_SPECS.keys())
    results: Dict[str, Any] = {}
    
    for p in platforms:
        if p in PLATFORM_SPECS:
            resized_pil = ImageProcessingService.resize_for_platform(img, p, mode=req.mode or "smart_pad")
            saved_id = storage_manager.save_image(resized_pil, category="generations", ext="png")
            results[p] = {
                "dimensions": PLATFORM_SPECS[p],
                "image_id": saved_id,
                "image_url": f"/api/images/{saved_id}"
            }
            
    return {
        "status": "COMPLETED",
        "variations": results
    }

@router.post("/localize")
async def localize_creative_text(req: CreativeLocalizeRequest):
    """
    Translates marketing headlines into English, Hindi, and Telugu,
    and applies typography reflow bounds to prevent visual overflow.
    """
    translations = []
    for text in req.texts:
        translated = LocalizationService.translate(text, req.target_lang)
        font_size = LocalizationService.compute_typography_reflow(
            translated,
            base_font_size=req.base_font_size or 36,
            max_box_width=req.max_box_width or 400
        )
        translations.append({
            "original_text": text,
            "translated_text": translated,
            "target_lang": req.target_lang,
            "fitted_font_size": font_size
        })
        
    return {
        "status": "COMPLETED",
        "target_lang": req.target_lang,
        "results": translations
    }
