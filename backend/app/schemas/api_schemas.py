from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ImageUploadResponse(BaseModel):
    image_id: str
    image_url: str
    width: int
    height: int
    format: str

class AIGenerateRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 1.0
    reference_image_id: Optional[str] = None

class AIEditRequest(BaseModel):
    image_id: str
    instruction_prompt: str
    mask_image_id: Optional[str] = None
    num_inference_steps: Optional[int] = 4

class AISegmentRequest(BaseModel):
    image_id: str
    points: List[List[float]] # [[x, y], ...]
    labels: List[int]         # [1, 0, ...]
    box: Optional[List[float]] = None

class AIRemoveBackgroundRequest(BaseModel):
    image_id: str

class AIReplaceBackgroundRequest(BaseModel):
    foreground_image_id: str
    background_prompt: str
    num_inference_steps: Optional[int] = 4

class CreativeResizeRequest(BaseModel):
    image_id: str
    platforms: Optional[List[str]] = None
    mode: Optional[str] = "smart_pad"

class CreativeLocalizeRequest(BaseModel):
    texts: List[str]
    target_lang: str # "en", "hi", "te"
    base_font_size: Optional[int] = 36
    max_box_width: Optional[int] = 400

class CampaignMatrixRequest(BaseModel):
    image_id: str
    user_prompt: Optional[str] = "Create a premium commercial advertisement. Professional photography lighting, preserve product identity."
    background_style: Optional[str] = "Luxury Studio"
    platforms: Optional[List[str]] = ["instagram_square", "instagram_portrait", "instagram_story", "youtube_thumbnail"]
    num_variations: Optional[int] = 4
    target_lang: Optional[str] = "en"
    headline: Optional[str] = "Glow Naturally, Every Day."
    cta: Optional[str] = "Shop Now"
    brand_name: Optional[str] = None
    primary_color: Optional[str] = "#6366f1"

class JobResponse(BaseModel):
    job_id: str
    task_type: str
    status: str
    progress: int
    result_image_id: Optional[str] = None
    result_image_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
