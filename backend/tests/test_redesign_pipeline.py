from PIL import Image, ImageDraw
from app.core.storage import storage_manager
from app.services.ocr_service import OCRService
from app.services.image_understanding_service import ImageUnderstandingService
from app.services.redesign_pipeline_service import RedesignPipelineService

def test_ai_image_redesign_pipeline():
    # 1. Create a mock crowded poster with headline, subtitle, and CTA
    test_poster = Image.new("RGB", (700, 700), (220, 220, 230))
    d = ImageDraw.Draw(test_poster)
    # Background graphics
    d.rectangle([100, 100, 600, 600], fill=(240, 240, 245))
    d.ellipse([250, 200, 450, 450], fill=(234, 88, 12)) # Product circle
    # Text
    d.text((80, 50), "MEGA ANNUAL TECH EXPO", fill=(0, 0, 0))
    d.text((80, 80), "Join 500+ Innovation Leaders This Weekend", fill=(50, 50, 50))
    d.text((80, 630), "Register Now: expo.tech/2026", fill=(255, 255, 255))
    
    img_id = storage_manager.save_image(test_poster, category="images")

    # 2. Test OCR Text Extraction
    text_items = OCRService.extract_text(test_poster)
    assert len(text_items) >= 2
    assert any("headline" in item.get("type", "") for item in text_items)

    # 3. Test Vision-Language Design Understanding
    audit = ImageUnderstandingService.analyze_design(test_poster, text_items)
    assert "audit_scores" in audit
    assert "identified_flaws" in audit
    assert "redesign_directions" in audit
    assert len(audit["redesign_directions"]) == 3
    assert "Direction_01_Modern_Minimalist" in audit["redesign_directions"]

    # 4. Test Complete End-to-End Redesign Execution
    redesign_result = RedesignPipelineService.execute_redesign(
        image_id=img_id,
        user_instructions="Redesign this image with a modern and professional appearance. Improve typography, spacing, lighting and layout."
    )

    assert redesign_result["original_image_id"] == img_id
    assert "extracted_text_elements" in redesign_result
    assert "redesigned_variations" in redesign_result
    assert len(redesign_result["redesigned_variations"]) == 3
    
    for dir_key, dir_data in redesign_result["redesigned_variations"].items():
        assert "image_url" in dir_data
        assert "title" in dir_data
        assert "color_palette" in dir_data
        # Verify reconstructed image exists on disk
        path = storage_manager.get_image_path(dir_data["image_id"])
        assert path is not None
