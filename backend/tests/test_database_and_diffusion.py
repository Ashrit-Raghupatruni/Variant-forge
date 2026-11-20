import pytest
from app.models.entities import ProjectRepository, VariantRepository, AnalysisRepository
from app.services.model_manager import model_manager, DiffusionArchitectureInfo
from app.services.controlnet_service import ControlNetService
from PIL import Image

def test_project_crud():
    proj = ProjectRepository.create({
        "name": "Nike Air Campaign",
        "brand_name": "Nike",
        "category": "footwear"
    })
    assert proj["id"] is not None
    assert proj["name"] == "Nike Air Campaign"

    fetched = ProjectRepository.get(proj["id"])
    assert fetched["brand_name"] == "Nike"

    updated = ProjectRepository.update(proj["id"], {"name": "Nike Air Max 2026"})
    assert updated["name"] == "Nike Air Max 2026"

    # Variant creation
    var = VariantRepository.create({
        "project_id": proj["id"],
        "platform": "instagram_square",
        "width": 1080,
        "height": 1080,
        "design_style": "Minimalist Luxury",
        "generation_model": "SDXL Inpaint (DDPM)"
    })
    assert var["id"] is not None

    variants = VariantRepository.list_by_project(proj["id"])
    assert len(variants) == 1

    # Cleanup
    assert ProjectRepository.delete(proj["id"]) is True

def test_diffusion_and_ddpm_architecture():
    summary = DiffusionArchitectureInfo.get_summary()
    assert "Denoising Diffusion Probabilistic Models (DDPM)" in summary["paradigm"]
    assert "GAN" in summary["comparison"]
    assert "VAE" in summary["comparison"]
    assert "forward_diffusion" in summary["processes"]
    assert "reverse_diffusion" in summary["processes"]
    assert "encoder" in summary["unet_architecture"]
    assert "decoder" in summary["unet_architecture"]
    assert "skip_connections" in summary["unet_architecture"]
    assert "timestep_embeddings" in summary["unet_architecture"]

def test_canny_structural_preservation():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    canny = ControlNetService.extract_canny(img)
    assert canny.size == (200, 200)
