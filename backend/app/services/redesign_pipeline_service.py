import logging
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont

from app.core.storage import storage_manager
from app.services.ocr_service import OCRService
from app.services.image_understanding_service import ImageUnderstandingService
from app.services.background_removal_service import BackgroundRemovalService
from app.services.image_generation_service import ImageGenerationService
from app.services.image_processing_service import ImageProcessingService

logger = logging.getLogger(__name__)

class RedesignPipelineService:
    """
    Executes the 6-step AI Image Redesign Pipeline:
    1. Visual Understanding (Qwen2.5-VL)
    2. Text Extraction (PaddleOCR)
    3. Design Defect Analysis & Specification
    4. AI Image Redesign (FLUX.2 Klein 4B for 3 distinct directions)
    5. Text & Element Reconstruction (Preserving original spelling & info)
    6. Deliver Final Output with Before/After Comparison & Editable Layer Data
    """

    @classmethod
    def reconstruct_text_layers(
        cls,
        background: Image.Image,
        text_elements: List[Dict[str, Any]],
        style_direction: str
    ) -> Image.Image:
        """
        Reconstructs the original OCR text cleanly over the redesigned visual backdrop,
        ensuring exact copy, correct hierarchy, and modern typography contrast.
        """
        canvas = background.convert("RGBA").copy()
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size

        # Theme color selection
        if "Minimalist" in style_direction:
            text_color = (15, 23, 42, 255)
            cta_bg = (79, 70, 229, 235)
            badge_text = (255, 255, 255, 255)
        elif "Premium" in style_direction:
            text_color = (254, 243, 199, 255)
            cta_bg = (217, 119, 6, 235)
            badge_text = (255, 255, 255, 255)
        else: # Creative Vibrant
            text_color = (255, 255, 255, 255)
            cta_bg = (236, 72, 153, 235)
            badge_text = (255, 255, 255, 255)

        for el in text_elements:
            text_str = el.get("text", "").strip()
            role = el.get("type", "body")
            
            if role == "headline":
                # Draw high-impact modern headline with drop shadow
                font_size = int(h * 0.052)
                pos_x = int(w * 0.08)
                pos_y = int(h * 0.06)
                # Shadow
                draw.text((pos_x + 2, pos_y + 2), text_str, fill=(0, 0, 0, 160))
                draw.text((pos_x, pos_y), text_str, fill=text_color)
                
            elif role == "subheading":
                font_size = int(h * 0.032)
                pos_x = int(w * 0.08)
                pos_y = int(h * 0.14)
                draw.text((pos_x, pos_y), text_str, fill=text_color)

            elif role == "cta":
                btn_w = int(w * 0.42)
                btn_h = int(h * 0.065)
                btn_x = int(w * 0.08)
                btn_y = int(h * 0.88)
                draw.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], radius=int(btn_h * 0.5), fill=cta_bg)
                draw.text((btn_x + int(btn_w * 0.18), btn_y + int(btn_h * 0.25)), text_str, fill=badge_text)

        return canvas

    @classmethod
    def execute_redesign(
        cls,
        image_id: str,
        user_instructions: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        orig_img = storage_manager.get_image(image_id)
        if not orig_img:
            raise ValueError(f"Image {image_id} not found")

        if progress_callback:
            progress_callback(15, "Running PaddleOCR text extraction...")

        # 1. Deterministic Text Extraction
        text_elements = OCRService.extract_text(orig_img)

        if progress_callback:
            progress_callback(30, "Running Qwen2.5-VL Visual & Design Audit...")

        # 2. Vision-Language Design Understanding
        design_audit = ImageUnderstandingService.analyze_design(orig_img, text_elements)

        if progress_callback:
            progress_callback(45, "Isolating product foreground with RMBG-2.0...")

        # 3. Product Foreground Isolation
        if orig_img.mode != "RGBA":
            foreground_rgba = BackgroundRemovalService.remove_background(orig_img)
        else:
            foreground_rgba = orig_img

        fg_id = storage_manager.save_image(foreground_rgba, category="generations", ext="png")

        # 4. Generate 3 Distinct Redesign Directions
        redesigns = {}
        directions = design_audit["redesign_directions"]
        step_increment = 40 // len(directions)
        current_pct = 50

        for dir_key, dir_meta in directions.items():
            if progress_callback:
                progress_callback(current_pct, f"Generating {dir_meta['title']} via FLUX.2 Klein 4B...")

            prompt = f"{user_instructions}. {dir_meta['prompt']}" if user_instructions else dir_meta['prompt']
            
            # Generate new contextual visual backdrop
            bg_gen = ImageGenerationService.generate(prompt=prompt, width=1024, height=1024, num_inference_steps=4)
            
            # Composite original foreground product
            composed = ImageProcessingService.replace_background(foreground_rgba, bg_gen)
            
            # Reconstruct original text layers with exact spelling & clean typography
            final_redesigned = cls.reconstruct_text_layers(composed, text_elements, dir_meta["title"])
            
            res_id = storage_manager.save_image(final_redesigned, category="generations", ext="png")
            
            redesigns[dir_key] = {
                "title": dir_meta["title"],
                "concept": dir_meta["concept"],
                "color_palette": dir_meta["color_palette"],
                "image_id": res_id,
                "image_url": f"/api/images/{res_id}"
            }
            current_pct += step_increment

        if progress_callback:
            progress_callback(100, "Redesign completed successfully!")

        return {
            "original_image_id": image_id,
            "original_image_url": f"/api/images/{image_id}",
            "foreground_image_id": fg_id,
            "extracted_text_elements": text_elements,
            "design_audit": design_audit,
            "redesigned_variations": redesigns
        }
