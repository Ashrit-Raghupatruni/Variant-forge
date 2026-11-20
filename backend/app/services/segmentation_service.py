import io
import base64
import logging
import requests
from typing import List, Optional, Tuple
from PIL import Image
from app.services.model_manager import model_manager
from app.ml.segmentation_runners import sam2_runner

logger = logging.getLogger(__name__)

class SegmentationService:
    """Promptable interactive object segmentation service using Meta SAM 2."""

    @staticmethod
    def segment(
        image: Image.Image,
        points: List[List[float]],
        labels: List[int],
        box: Optional[List[float]] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Takes image and click prompts.
        Returns: (isolated_rgba_cutout, binary_mask)
        """
        # Route 1: Remote GPU Worker
        if model_manager.is_remote_enabled():
            try:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                resp = requests.post(
                    f"{model_manager.remote_url}/segment",
                    json={"image_base64": img_b64, "points": points, "labels": labels, "box": box},
                    timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    cutout_b64 = data["cutout_base64"].split(",", 1)[-1]
                    mask_b64 = data["mask_base64"].split(",", 1)[-1]
                    cutout = Image.open(io.BytesIO(base64.b64decode(cutout_b64)))
                    mask = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    return cutout, mask
            except Exception as e:
                logger.warning(f"Remote SAM 2 call failed, falling back to local: {e}")

        # Route 2: Local SAM 2 runner
        return sam2_runner.segment_with_prompts(image=image, points=points, labels=labels, box=box)
