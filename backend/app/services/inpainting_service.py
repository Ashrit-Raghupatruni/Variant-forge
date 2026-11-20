import os
import io
import base64
import logging
import requests
import numpy as np
from PIL import Image
from typing import Optional

from app.services.model_manager import model_manager
from app.config import settings

logger = logging.getLogger(__name__)

class InpaintingService:
    """
    Executes SDXL Inpainting based on DDPM Reverse Diffusion:
    Denoises masked regions while strictly preserving unmasked creative content.
    Routes to remote Colab GPU worker or executes locally when available.
    """
    @staticmethod
    def inpaint(
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, artifacts, distorted text",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.95,
        seed: Optional[int] = 42
    ) -> Image.Image:
        # Pre-process image & mask
        w, h = image.size
        # Standardize dimensions to multiples of 8 for U-Net compatibility
        w_clean = (w // 8) * 8
        h_clean = (h // 8) * 8
        image_rgb = image.resize((w_clean, h_clean)).convert("RGB")
        mask_l = mask.resize((w_clean, h_clean)).convert("L")

        # 1. Check if Remote GPU Worker (Colab) is available
        if model_manager.is_remote_enabled():
            try:
                logger.info(f"Dispatching Inpainting to remote GPU: {model_manager.remote_url}/inpaint")
                buf_img = io.BytesIO()
                image_rgb.save(buf_img, format="PNG")
                b64_img = base64.b64encode(buf_img.getvalue()).decode("utf-8")

                buf_mask = io.BytesIO()
                mask_l.save(buf_mask, format="PNG")
                b64_mask = base64.b64encode(buf_mask.getvalue()).decode("utf-8")

                payload = {
                    "image_base64": b64_img,
                    "mask_base64": b64_mask,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed
                }
                resp = requests.post(f"{model_manager.remote_url}/inpaint", json=payload, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    res_bytes = base64.b64decode(data["image_base64"])
                    return Image.open(io.BytesIO(res_bytes)).resize((w, h))
                else:
                    logger.warning(f"Remote inpainting error: {resp.text}")
            except Exception as e:
                logger.warning(f"Remote inpainting worker connection failed: {e}")

        # 2. Local GPU execution if available
        pipe = model_manager.load_sdxl_inpaint()
        if pipe is not None:
            try:
                import torch
                generator = torch.Generator(device=model_manager.device).manual_seed(seed) if seed is not None else None
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image_rgb,
                    mask_image=mask_l,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator
                ).images[0]
                return result.resize((w, h))
            except Exception as e:
                logger.error(f"Local inpainting execution error: {e}")

        # 3. Fail clearly if neither remote GPU nor local CUDA is available
        logger.error("No active GPU found for SDXL Inpainting.")
        raise RuntimeError("GPU Not Connected: SDXL Inpainting requires an active Colab GPU or local CUDA.")
