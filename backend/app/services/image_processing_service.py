from typing import Dict, Tuple, List, Optional
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# Official platform specifications
PLATFORM_SPECS: Dict[str, Tuple[int, int]] = {
    "instagram_square": (1080, 1080),   # 1:1
    "instagram_portrait": (1080, 1350), # 4:5
    "instagram_story": (1080, 1920),    # 9:16
    "youtube_thumbnail": (1280, 720),   # 16:9
    "linkedin_landscape": (1200, 627),  # 1.91:1
}

class ImageProcessingService:
    """
    OpenCV and Pillow image processing engine:
    - Exact multi-platform aspect ratio conversion & smart padding
    - Transparent foreground background replacement and shadow composition
    - Mask processing and alpha matting
    """
    @staticmethod
    def resize_for_platform(
        image: Image.Image,
        platform_key: str,
        mode: str = "smart_pad"  # "smart_pad", "crop_fill", "fit_center"
    ) -> Image.Image:
        if platform_key not in PLATFORM_SPECS:
            raise ValueError(f"Unknown platform '{platform_key}'. Valid: {list(PLATFORM_SPECS.keys())}")
        
        target_w, target_h = PLATFORM_SPECS[platform_key]
        orig_w, orig_h = image.size

        if mode == "crop_fill":
            return ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.LANCZOS)

        elif mode == "smart_pad":
            # Smart background expansion: zoomed blurred backdrop
            bg = ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
            
            # Scale foreground gracefully to retain visual prominence
            scale = min((target_w * 0.85) / orig_w, (target_h * 0.85) / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            fg = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            
            if fg.mode == "RGBA":
                bg.paste(fg, (offset_x, offset_y), fg)
            else:
                bg.paste(fg, (offset_x, offset_y))
            return bg

        else: # fit_center
            canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            fg = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            canvas.paste(fg, (offset_x, offset_y), fg if fg.mode == "RGBA" else None)
            return canvas

    @staticmethod
    def replace_background(foreground_rgba: Image.Image, background: Image.Image) -> Image.Image:
        """Composites an isolated RGBA product onto a newly generated background."""
        w, h = background.size
        fg_scaled = ImageOps.contain(foreground_rgba, (int(w * 0.75), int(h * 0.75)), method=Image.Resampling.LANCZOS)
        
        # Calculate centered placement
        offset_x = (w - fg_scaled.width) // 2
        offset_y = (h - fg_scaled.height) // 2 + int(h * 0.04)

        # Soft drop shadow
        shadow = Image.new("RGBA", (fg_scaled.width, int(fg_scaled.height * 0.2)), (0, 0, 0, 0))
        from PIL import ImageDraw
        d = ImageDraw.Draw(shadow)
        d.ellipse([int(fg_scaled.width * 0.1), 0, int(fg_scaled.width * 0.9), int(fg_scaled.height * 0.18)], fill=(0, 0, 0, 110))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))

        composite = background.convert("RGBA").copy()
        composite.paste(shadow, (offset_x, offset_y + fg_scaled.height - int(fg_scaled.height * 0.1)), shadow)
        composite.paste(fg_scaled, (offset_x, offset_y), fg_scaled)
        return composite
