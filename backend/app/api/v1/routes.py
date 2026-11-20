import os
import io
import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

from app.ml.flux_runner import flux_runner
from app.ml.segmentation_runners import rmbg_runner, sam2_runner
from app.services.resizer_service import CreativeResizer, PLATFORM_SPECS
from app.services.localization_service import LocalizationService
from app.config import settings

router = APIRouter()

class GenerationRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 4

class ResizeRequest(BaseModel):
    image_base64: str
    platforms: Optional[List[str]] = None
    mode: Optional[str] = "smart_pad"

class TranslateRequest(BaseModel):
    texts: List[str]
    target_lang: str

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes))

@router.post("/flux/generate")
async def generate_flux_image(req: GenerationRequest):
    """
    Image-to-Image editing or Text-to-Image generation using FLUX.2-klein-4B.
    """
    input_img = None
    if req.image_base64:
        input_img = base64_to_pil(req.image_base64)

    output_img = flux_runner.edit_or_generate(
        prompt=req.prompt,
        input_image=input_img,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps
    )
    return {
        "status": "success",
        "prompt": req.prompt,
        "image_base64": f"data:image/png;base64,{pil_to_base64(output_img)}"
    }

@router.post("/segment/remove-bg")
async def remove_background(image: UploadFile = File(...)):
    """
    Background isolation using RMBG 2.0.
    """
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents))
    result_img = rmbg_runner.remove_background(pil_img)
    return {
        "status": "success",
        "image_base64": f"data:image/png;base64,{pil_to_base64(result_img)}"
    }

class SAM2Request(BaseModel):
    image_base64: str
    points: List[List[float]] # [[x, y], ...]
    labels: List[int]         # [1, 0, ...]
    box: Optional[List[float]] = None

@router.post("/segment/sam2")
async def segment_with_sam2(req: SAM2Request):
    """
    Interactive object segmentation using SAM 2 Image Predictor.
    Accepts point prompts [x, y] with labels (1=foreground, 0=background) or bounding boxes.
    Returns the isolated transparent RGBA layer and binary mask.
    """
    pil_img = base64_to_pil(req.image_base64)
    cutout_img, mask_img = sam2_runner.segment_with_prompts(
        image=pil_img,
        points=req.points,
        labels=req.labels,
        box=req.box
    )
    return {
        "status": "success",
        "cutout_base64": f"data:image/png;base64,{pil_to_base64(cutout_img)}",
        "mask_base64": f"data:image/png;base64,{pil_to_base64(mask_img)}"
    }

@router.post("/resize/multi-platform")
async def resize_multi_platform(req: ResizeRequest):
    """
    Generate platform-specific creative variations (Instagram, YouTube, LinkedIn, etc.)
    """
    pil_img = base64_to_pil(req.image_base64)
    platforms = req.platforms or list(PLATFORM_SPECS.keys())
    variations = {}
    for p in platforms:
        if p in PLATFORM_SPECS:
            res_img = CreativeResizer.resize_for_platform(pil_img, p, mode=req.mode)
            variations[p] = {
                "dimensions": PLATFORM_SPECS[p],
                "image_base64": f"data:image/png;base64,{pil_to_base64(res_img)}"
            }
    return {
        "status": "success",
        "variations": variations
    }

@router.post("/localization/translate")
async def translate_copy(req: TranslateRequest):
    """
    Localize marketing copy and compute dynamic typography reflow sizing.
    """
    results = []
    for text in req.texts:
        translated = LocalizationService.translate_text(text, req.target_lang)
        fitted_font_size = LocalizationService.calculate_typography_reflow(translated, original_font_size=36, target_box_width=400)
        results.append({
            "original": text,
            "translated": translated,
            "target_lang": req.target_lang,
            "suggested_font_size": fitted_font_size
        })
    return {
        "status": "success",
        "translations": results
    }
