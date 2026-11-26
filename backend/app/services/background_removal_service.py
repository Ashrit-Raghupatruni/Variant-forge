import io
import base64
import logging
import requests
from PIL import Image
from app.services.model_manager import model_manager
from app.ml.segmentation_runners import rmbg_runner

logger = logging.getLogger(__name__)

class BackgroundRemovalService:
    """Background removal and alpha-transparency extraction using BRIA RMBG-2.0."""

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """
        Takes RGB/RGBA image, strips background, returns RGBA image with transparent alpha channel.
        """
        # Route 1: Managed Cloud AI (BiRefNet via Fal.ai or local rembg)
        from app.services.cloud_ai_service import CloudAIService
        if CloudAIService.get_fal_key():
            try:
                return CloudAIService.remove_background(image)
            except Exception as e:
                logger.warning(f"Cloud background removal failed: {e}. Trying remote GPU...")

        # Route 2: Remote GPU Worker
        if model_manager.is_remote_enabled():
            try:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                resp = requests.post(
                    f"{model_manager.remote_url}/remove-bg",
                    json={"image_base64": img_b64},
                    timeout=180
                )
                if resp.status_code == 200:
                    data = resp.json()
                    b64_str = data["image_base64"].split(",", 1)[-1]
                    return Image.open(io.BytesIO(base64.b64decode(b64_str)))
            except Exception as e:
                logger.warning(f"Remote RMBG call failed: {e}")

        # Route 2: Local runner (PyTorch AutoModel if loaded)
        if rmbg_runner.model is not None:
            return rmbg_runner.remove_background(image)

        raise RuntimeError("GPU Not Connected: Background Removal (RMBG-2.0) requires an active Colab GPU or local CUDA.")
