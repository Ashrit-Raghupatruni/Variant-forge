import os
import zipfile
from PIL import Image
from app.core.storage import storage_manager
from app.services.campaign_pipeline_service import CampaignPipelineService

def test_campaign_matrix_execution():
    # 1. Create a synthetic test product image (e.g. serum bottle with white background)
    test_img = Image.new("RGB", (600, 600), (250, 250, 250))
    # Draw bottle shape in center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([220, 150, 380, 480], fill=(217, 119, 6)) # Amber glass
    draw.rectangle([260, 90, 340, 150], fill=(30, 41, 59))   # Dropper cap
    draw.rectangle([240, 250, 360, 380], fill=(255, 255, 255)) # Label
    
    img_id = storage_manager.save_image(test_img, category="images")

    # 2. Execute 4 variations x 4 platforms = 16 creative files
    platforms = ["instagram_square", "instagram_portrait", "instagram_story", "youtube_thumbnail"]
    result = CampaignPipelineService.execute_campaign(
        image_id=img_id,
        user_prompt="Create a premium skincare commercial advertisement with luxury lighting",
        platforms=platforms,
        num_variations=4,
        target_lang="en",
        headline="Glow Naturally, Every Day.",
        cta="Shop Now"
    )

    # 3. Assert matrix completeness
    assert result["num_variations"] == 4
    assert result["total_images_generated"] == 16
    assert len(result["variations"]) == 4

    # Verify each style contains all 4 requested platform dimensions
    for style_key, style_data in result["variations"].items():
        assert len(style_data["platforms"]) == 4
        assert "instagram_square" in style_data["platforms"]
        assert "instagram_story" in style_data["platforms"]
        assert style_data["platforms"]["instagram_square"]["dimensions"] == (1080, 1080)
        assert style_data["platforms"]["instagram_story"]["dimensions"] == (1080, 1920)

    # 4. Assert ZIP archive contains the exact requested folder hierarchy
    zip_path = storage_manager.get_image_path(result["zip_id"])
    assert zip_path is not None and os.path.exists(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert len(names) == 16
        assert "VariantForge_Output/Variation_01_Luxury/Instagram_Square.png" in names
        assert "VariantForge_Output/Variation_02_Minimalist/Instagram_Story.png" in names
        assert "VariantForge_Output/Variation_03_Nature/YouTube_Thumbnail.png" in names
        assert "VariantForge_Output/Variation_04_Premium/Instagram_Portrait.png" in names
