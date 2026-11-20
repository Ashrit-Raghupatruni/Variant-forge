from PIL import Image
from app.services.image_processing_service import ImageProcessingService, PLATFORM_SPECS
from app.services.localization_service import LocalizationService
from app.services.model_manager import model_manager
from app.core.storage import storage_manager
from app.core.job_queue import job_registry, JobStatus

def test_platform_specs_exist():
    assert "instagram_square" in PLATFORM_SPECS
    assert "instagram_story" in PLATFORM_SPECS
    assert "youtube_thumbnail" in PLATFORM_SPECS
    assert PLATFORM_SPECS["instagram_square"] == (1080, 1080)
    assert PLATFORM_SPECS["instagram_story"] == (1080, 1920)

def test_image_processing_resize():
    test_img = Image.new("RGB", (500, 500), color=(255, 0, 0))
    resized_ig = ImageProcessingService.resize_for_platform(test_img, "instagram_square")
    assert resized_ig.size == (1080, 1080)
    
    resized_story = ImageProcessingService.resize_for_platform(test_img, "instagram_story")
    assert resized_story.size == (1080, 1920)

def test_localization_and_reflow_hindi_telugu():
    # Test Hindi
    hi_trans = LocalizationService.translate("New Arrival", "hi")
    assert hi_trans == "नया आगमन"
    
    # Test Telugu
    te_trans = LocalizationService.translate("Shop Now", "te")
    assert te_trans == "ఇప్పుడే కొనండి"
    
    # Test Typography Reflow
    reflow_size = LocalizationService.compute_typography_reflow("లభించే అత్యుత్తమ ఆఫర్లు మరియు డీల్స్", base_font_size=40, max_box_width=200)
    assert reflow_size < 40

def test_storage_and_jobs():
    test_img = Image.new("RGB", (100, 100), (0, 255, 0))
    img_id = storage_manager.save_image(test_img)
    loaded = storage_manager.get_image(img_id)
    assert loaded is not None
    assert loaded.size == (100, 100)

    job_id = job_registry.create_job("TEST_JOB")
    job_registry.update_job(job_id, status=JobStatus.COMPLETED, progress=100)
    job = job_registry.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED
    assert job["progress"] == 100
