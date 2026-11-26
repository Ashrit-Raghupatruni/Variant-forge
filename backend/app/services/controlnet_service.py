import io
import base64
import logging
import requests
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional

from app.services.model_manager import model_manager
from app.config import settings

logger = logging.getLogger(__name__)

class ControlNetService:
    """
    ControlNet structural preservation service:
    Extracts Canny edge maps or Depth maps from the source campaign creative
    and conditions the diffusion U-Net skip connections to preserve layout geometry.
    """
    @staticmethod
    def extract_canny(
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """Generates a Canny edge structural control map with PIL fallback."""
        try:
            import cv2
            img_np = np.array(image.convert("RGB"))
            edges = cv2.Canny(img_np, low_threshold, high_threshold)
            edges_rgb = np.stack([edges] * 3, axis=-1)
            return Image.fromarray(edges_rgb)
        except ImportError:
            # Fallback edge filter when cv2 is not installed locally
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return edges.convert("RGB")

    @staticmethod
    def generate_controlled(
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, artifacts, distorted text",
        conditioning_scale: float = 0.8,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 42
    ) -> Image.Image:
        w, h = image.size
        w_clean = (w // 8) * 8
        h_clean = (h // 8) * 8
        canny_img = ControlNetService.extract_canny(image.resize((w_clean, h_clean)))

        # 1. Dispatch to Managed Cloud AI (Fal.ai ControlNet Canny) if available
        from app.services.cloud_ai_service import CloudAIService
        if CloudAIService.get_fal_key():
            try:
                return CloudAIService.controlnet_canny(
                    image=image,
                    canny_map=canny_img,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    conditioning_scale=conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            except Exception as e:
                logger.warning(f"Cloud ControlNet failed: {e}. Trying remote Colab worker...")

        # 2. Dispatch to Remote Worker if available
        if model_manager.is_remote_enabled():
            try:
                buf_img = io.BytesIO()
                image.save(buf_img, format="PNG")
                b64_img = base64.b64encode(buf_img.getvalue()).decode("utf-8")

                buf_canny = io.BytesIO()
                canny_img.save(buf_canny, format="PNG")
                b64_canny = base64.b64encode(buf_canny.getvalue()).decode("utf-8")

                payload = {
                    "image_base64": b64_img,
                    "canny_base64": b64_canny,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "conditioning_scale": conditioning_scale,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                }
                resp = requests.post(f"{model_manager.remote_url}/controlnet", json=payload, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    res_bytes = base64.b64decode(data["image_base64"])
                    return Image.open(io.BytesIO(res_bytes)).resize((w, h))
            except Exception as e:
                logger.warning(f"Remote ControlNet request failed: {e}")

        # 2. Local GPU ControlNet execution
        pipe = model_manager.load_controlnet_canny()
        if pipe is not None:
            try:
                import torch
                generator = torch.Generator(device=model_manager.device).manual_seed(seed) if seed is not None else None
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=canny_img,
                    controlnet_conditioning_scale=conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                return result.resize((w, h))
            except Exception as e:
                logger.error(f"Local ControlNet execution failed: {e}")

        # 3. Fail clearly if neither remote GPU nor local CUDA is available
        logger.error("No active GPU found for ControlNet SDXL.")
        raise RuntimeError("GPU Not Connected: ControlNet SDXL requires an active Colab GPU or local CUDA.")
