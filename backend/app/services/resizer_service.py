from typing import Dict, Tuple
from PIL import Image, ImageOps, ImageFilter

PLATFORM_SPECS: Dict[str, Tuple[int, int]] = {
    "instagram_post": (1080, 1080),      # 1:1 Square
    "instagram_story": (1080, 1920),     # 9:16 Vertical
    "linkedin_banner": (1200, 628),      # 1.91:1 Landscape
    "youtube_thumbnail": (1280, 720),    # 16:9 Landscape
    "twitter_post": (1200, 675),         # 16:9
    "facebook_feed": (1080, 1350),       # 4:5 Portrait
}

class CreativeResizer:
    """
    Automated multi-platform resizing & composition service.
    Preserves foreground product placement and applies smart background expansion/blur padding.
    """
    @staticmethod
    def resize_for_platform(
        image: Image.Image,
        platform_key: str,
        mode: str = "smart_pad" # "smart_pad", "fit_center", "crop_fill"
    ) -> Image.Image:
        if platform_key not in PLATFORM_SPECS:
            raise ValueError(f"Unknown platform key: {platform_key}. Available: {list(PLATFORM_SPECS.keys())}")
        
        target_w, target_h = PLATFORM_SPECS[platform_key]
        orig_w, orig_h = image.size
        
        if mode == "crop_fill":
            return ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.LANCZOS)
        
        elif mode == "smart_pad":
            # Smart background expansion: scaled blurred background + sharp centered subject
            bg = ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=25))
            
            # Scale foreground to fit safely within bounds
            scale = min((target_w * 0.85) / orig_w, (target_h * 0.85) / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            fg = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Composite foreground over blurred background
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            
            if fg.mode == "RGBA":
                bg.paste(fg, (offset_x, offset_y), fg)
            else:
                bg.paste(fg, (offset_x, offset_y))
            return bg

        else: # fit_center with solid/gradient fill
            canvas = Image.new("RGBA", (target_w, target_h), (245, 247, 250, 255))
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            fg = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            canvas.paste(fg, (offset_x, offset_y), fg if fg.mode == "RGBA" else None)
            return canvas

    @staticmethod
    def generate_all_variations(image: Image.Image) -> Dict[str, Image.Image]:
        return {
            platform: CreativeResizer.resize_for_platform(image, platform)
            for platform in PLATFORM_SPECS.keys()
        }
