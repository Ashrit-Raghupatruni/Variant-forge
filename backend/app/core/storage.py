import os
import uuid
import shutil
from typing import Optional, Dict
from PIL import Image

class StorageManager:
    """Manages persistent image uploads, masks, and generated creatives."""
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
        self.base_dir = base_dir
        self.images_dir = os.path.join(self.base_dir, "images")
        self.masks_dir = os.path.join(self.base_dir, "masks")
        self.generations_dir = os.path.join(self.base_dir, "generations")
        
        for d in [self.images_dir, self.masks_dir, self.generations_dir]:
            os.makedirs(d, exist_ok=True)
            
        self._index: Dict[str, str] = {}

    def save_image(self, image: Image.Image, category: str = "images", ext: str = "png") -> str:
        """Saves a PIL image and returns an image_id."""
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.{ext}"
        target_dir = getattr(self, f"{category}_dir", self.images_dir)
        filepath = os.path.join(target_dir, filename)
        
        # Save as PNG if RGBA or specified
        if image.mode == "RGBA" or ext.lower() == "png":
            image.save(filepath, format="PNG")
        else:
            image.convert("RGB").save(filepath, format="JPEG", quality=95)
            
        self._index[image_id] = filepath
        return image_id

    def save_bytes(self, content: bytes, ext: str = "png", category: str = "images") -> str:
        """Saves raw byte content from file uploads."""
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.{ext}"
        target_dir = getattr(self, f"{category}_dir", self.images_dir)
        filepath = os.path.join(target_dir, filename)
        with open(filepath, "wb") as f:
            f.write(content)
        self._index[image_id] = filepath
        return image_id

    def get_image_path(self, image_id: str) -> Optional[str]:
        if image_id in self._index and os.path.exists(self._index[image_id]):
            return self._index[image_id]
        # Search directories
        for folder in [self.images_dir, self.generations_dir, self.masks_dir]:
            for ext in ["png", "jpg", "jpeg", "webp"]:
                candidate = os.path.join(folder, f"{image_id}.{ext}")
                if os.path.exists(candidate):
                    self._index[image_id] = candidate
                    return candidate
        return None

    def get_image(self, image_id: str) -> Optional[Image.Image]:
        path = self.get_image_path(image_id)
        if path and os.path.exists(path):
            return Image.open(path)
        return None

    def get_mask(self, mask_id: str) -> Optional[Image.Image]:
        """Retrieves a mask image by ID."""
        return self.get_image(mask_id)

storage_manager = StorageManager()
