import io
import base64
import logging
import requests
from typing import Optional
from PIL import Image
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)

class ImageGenerationService:
    """Text-to-image generation service using FLUX.2 Klein 4B."""

    @staticmethod
    def generate(
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Image.Image:
        # Route 1: Remote GPU Worker (Google Colab / Dedicated Cloud GPU)
        if model_manager.is_remote_enabled():
            try:
                logger.info(f"Routing FLUX.2 generation to remote worker at {model_manager.remote_url}")
                resp = requests.post(
                    f"{model_manager.remote_url}/generate",
                    json={
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                    },
                    timeout=300
                )
                if resp.status_code == 200:
                    data = resp.json()
                    b64_str = data["image_base64"].split(",", 1)[-1]
                    return Image.open(io.BytesIO(base64.b64decode(b64_str)))
                else:
                    logger.warning(f"Remote worker returned error: {resp.status_code}")
            except Exception as e:
                logger.warning(f"Remote generation call failed ({e}). Falling back to local generation.")

        # Route 2: Local GPU execution
        pipe = model_manager.load_flux()
        if pipe is not None:
            import torch
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            return output.images[0]

        # Route 3: Fail clearly if no GPU is connected
        raise RuntimeError("GPU Not Connected: FLUX.2 image generation requires an active Colab GPU or local CUDA.")
