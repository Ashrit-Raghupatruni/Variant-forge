import io
import base64
import logging
import requests
from typing import Optional
from PIL import Image
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)

class ImageEditingService:
    """Image-to-image and in-context creative editing service using FLUX.2 Klein 4B."""

    @staticmethod
    def edit(
        image: Image.Image,
        instruction_prompt: str,
        mask: Optional[Image.Image] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
    ) -> Image.Image:
        w, h = image.size

        # Route 1: Remote GPU Worker
        if model_manager.is_remote_enabled():
            logger.info("Routing FLUX.2 image editing to remote GPU worker")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            payload = {
                "prompt": instruction_prompt,
                "image_base64": img_b64,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            resp = requests.post(f"{model_manager.remote_url}/generate", json=payload, timeout=120)
            resp.raise_for_status()
            b64_str = resp.json()["image_base64"].split(",", 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64_str)))

        # Route 2: Local GPU execution
        pipe = model_manager.load_flux()
        if pipe is not None:
            output = pipe(
                image=image,
                prompt=instruction_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            return output.images[0]

        # Route 3: Local pipeline runner
        from app.ml.flux_runner import flux_runner
        return flux_runner.edit_or_generate(
            prompt=instruction_prompt,
            input_image=image,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps
        )
