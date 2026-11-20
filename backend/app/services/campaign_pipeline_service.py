import os
import io
import zipfile
import logging
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw, ImageFont

from app.core.storage import storage_manager
from app.services.background_removal_service import BackgroundRemovalService
from app.services.image_generation_service import ImageGenerationService
from app.services.image_processing_service import ImageProcessingService, PLATFORM_SPECS
from app.services.localization_service import LocalizationService

logger = logging.getLogger(__name__)

STYLE_PROMPTS = {
    "Variation_01_Luxury": {
        "title": "Luxury Studio",
        "prompt": "soft beige luxury studio environment with elegant marble countertop, soft natural morning shadows, high-end commercial cosmetic product photography, shallow depth of field, warm diffused lighting, 8k"
    },
    "Variation_02_Minimalist": {
        "title": "Minimalist Studio",
        "prompt": "clean high-key minimalist white studio podium, crisp geometric lines, subtle gray shadows, modern Scandinavian commercial product layout, bright balanced lighting, ultra-sharp"
    },
    "Variation_03_Nature": {
        "title": "Botanical Nature",
        "prompt": "serene botanical garden setting, fresh green monstera leaves and eucalyptus branches, dappled sunlight filtering through leaves, organic raw wood surface, natural earthy aesthetic"
    },
    "Variation_04_Premium": {
        "title": "Premium Commercial",
        "prompt": "dramatic dark luxury cosmetics advertisement setup, matte obsidian reflective surface, cinematic rim lighting, subtle metallic accents, sophisticated high-fashion advertising aesthetic"
    }
}

PLATFORM_FILENAME_MAP = {
    "instagram_square": "Instagram_Square.png",
    "instagram_portrait": "Instagram_Portrait.png",
    "instagram_story": "Instagram_Story.png",
    "youtube_thumbnail": "YouTube_Thumbnail.png",
    "linkedin_landscape": "LinkedIn_Post.png"
}

class CampaignPipelineService:
    """
    Executes the end-to-end 7-step VariantForge AI Campaign Matrix Pipeline:
    1. Upload & Analysis
    2. Background Removal (RMBG-2.0 / SAM 2)
    3. AI Background Generation (FLUX.2 Klein 4B)
    4. Creative Variation Generation (4 distinct styles)
    5. Multi-Platform Optimization (16–20 platform files)
    6. Localization & Typography Overlay (English, Hindi, Telugu)
    7. Hierarchical ZIP Export (VariantForge_Output/)
    """

    @staticmethod
    def render_brand_overlays(
        base_img: Image.Image,
        headline: Optional[str] = None,
        cta: Optional[str] = None,
        target_lang: str = "en",
        primary_color: str = "#6366f1"
    ) -> Image.Image:
        """Draws localized campaign headline and call-to-action button onto the image."""
        if not headline and not cta:
            return base_img

        canvas = base_img.convert("RGBA").copy()
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size

        # 1. Headline Overlay with localization
        if headline:
            trans_headline = LocalizationService.translate(headline, target_lang)
            font_size = LocalizationService.compute_typography_reflow(trans_headline, base_font_size=int(h * 0.045), max_box_width=int(w * 0.8))
            
            # Draw subtle top gradient banner for legibility
            banner = Image.new("RGBA", (w, int(h * 0.22)), (0, 0, 0, 0))
            b_draw = ImageDraw.Draw(banner)
            for y in range(banner.height):
                alpha = int(140 * (1 - (y / banner.height)))
                b_draw.line([(0, y), (w, y)], fill=(15, 23, 42, alpha))
            canvas.paste(banner, (0, 0), banner)

            # Simple fallback rendering for font
            pos_x = int(w * 0.08)
            pos_y = int(h * 0.05)
            # Text shadow
            draw.text((pos_x + 2, pos_y + 2), trans_headline, fill=(0, 0, 0, 180))
            draw.text((pos_x, pos_y), trans_headline, fill=(255, 255, 255, 255))

        # 2. Call-To-Action (CTA) Button
        if cta:
            trans_cta = LocalizationService.translate(cta, target_lang)
            btn_w = int(w * 0.38)
            btn_h = int(h * 0.065)
            btn_x = int(w * 0.08)
            btn_y = int(h * 0.88)
            
            # Draw rounded button
            draw.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], radius=int(btn_h * 0.5), fill=(99, 102, 241, 235))
            # Center CTA text inside button
            draw.text((btn_x + int(btn_w * 0.22), btn_y + int(btn_h * 0.25)), trans_cta, fill=(255, 255, 255, 255))

        return canvas

    @classmethod
    def execute_campaign(
        cls,
        image_id: str,
        user_prompt: str,
        platforms: List[str],
        num_variations: int = 4,
        target_lang: str = "en",
        headline: Optional[str] = None,
        cta: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Runs the full 7-step pipeline and returns the complete asset matrix.
        """
        orig_img = storage_manager.get_image(image_id)
        if not orig_img:
            raise ValueError(f"Original image {image_id} not found")

        if progress_callback:
            progress_callback(10, "Extracting foreground with RMBG-2.0...")

        # Step 2: Background Removal
        if orig_img.mode != "RGBA":
            foreground_rgba = BackgroundRemovalService.remove_background(orig_img)
        else:
            foreground_rgba = orig_img

        fg_id = storage_manager.save_image(foreground_rgba, category="generations", ext="png")

        # Step 3 & 4: Generate Distinct Creative Variations
        selected_styles = list(STYLE_PROMPTS.keys())[:num_variations]
        variations_matrix: Dict[str, Any] = {}

        total_steps = len(selected_styles) * len(platforms)
        current_step = 0

        for style_key in selected_styles:
            style_info = STYLE_PROMPTS[style_key]
            combined_prompt = f"{user_prompt}. {style_info['prompt']}" if user_prompt else style_info['prompt']

            if progress_callback:
                progress_callback(
                    20 + int(current_step / total_steps * 60),
                    f"Generating {style_info['title']} variation with FLUX.2 Klein 4B..."
                )

            # Generate new contextual background
            bg_gen = ImageGenerationService.generate(prompt=combined_prompt, width=1024, height=1024, num_inference_steps=4)
            # Composite product over background with realistic drop shadow
            composite = ImageProcessingService.replace_background(foreground_rgba, bg_gen)

            # Step 6: Render localized brand headlines & CTA buttons
            styled_creative = cls.render_brand_overlays(
                composite, headline=headline, cta=cta, target_lang=target_lang
            )

            # Step 5: Multi-Platform Optimization (1:1, 4:5, 9:16, 16:9, etc.)
            platform_files = {}
            for p_key in platforms:
                if p_key in PLATFORM_SPECS:
                    resized_img = ImageProcessingService.resize_for_platform(styled_creative, p_key, mode="smart_pad")
                    img_id = storage_manager.save_image(resized_img, category="generations", ext="png")
                    platform_files[p_key] = {
                        "filename": PLATFORM_FILENAME_MAP.get(p_key, f"{p_key}.png"),
                        "dimensions": PLATFORM_SPECS[p_key],
                        "image_id": img_id,
                        "image_url": f"/api/images/{img_id}"
                    }
                    current_step += 1

            variations_matrix[style_key] = {
                "style_title": style_info["title"],
                "prompt": combined_prompt,
                "platforms": platform_files
            }

        if progress_callback:
            progress_callback(90, "Bundling campaign matrix into hierarchical ZIP archive...")

        # Step 7: Create Hierarchical ZIP Archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for style_key, s_data in variations_matrix.items():
                folder_name = style_key
                for p_key, p_data in s_data["platforms"].items():
                    file_path = storage_manager.get_image_path(p_data["image_id"])
                    if file_path and os.path.exists(file_path):
                        arc_path = f"VariantForge_Output/{folder_name}/{p_data['filename']}"
                        zip_file.write(file_path, arc_path)

        zip_bytes = zip_buffer.getvalue()
        zip_id = storage_manager.save_bytes(zip_bytes, ext="zip", category="generations")

        if progress_callback:
            progress_callback(100, "Campaign generation completed!")

        return {
            "foreground_image_id": fg_id,
            "foreground_url": f"/api/images/{fg_id}",
            "num_variations": len(selected_styles),
            "total_images_generated": len(selected_styles) * len(platforms),
            "target_lang": target_lang,
            "zip_id": zip_id,
            "zip_url": f"/api/campaign/export-zip/{zip_id}",
            "variations": variations_matrix
        }
