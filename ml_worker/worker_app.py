"""
VariantForge AI - Dedicated Colab T4 GPU Inference Worker
Engineered to run on Google Colab (Tesla T4 16GB VRAM).

Supports all 7 models using dynamic on-demand lifecycle to avoid OOM:
  1. SDXL Inpainting 1.0 (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
  2. SDXL ControlNet Canny (diffusers/controlnet-canny-sdxl-1.0)
  3. FLUX.2 Klein 4B (black-forest-labs/FLUX.2-klein-4B with CPU offload)
  4. Qwen2.5-VL 3B Instruct (Qwen/Qwen2.5-VL-3B-Instruct)
  5. BRIA RMBG-2.0 / 1.4 (AutoModelForImageSegmentation)
  6. Meta SAM 2.1 (SAM2ImagePredictor Hiera Large)
  7. PaddleOCR / EasyOCR (Text bounding box & copy extraction)
"""

import os
import io
import gc
import sys
import base64
import time
import torch
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(title="VariantForge AI - Remote T4 GPU Worker", version="3.0.0")

def cleanup_gpu():
    """Forces garbage collection and releases cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def b64_to_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def image_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ==============================================================================
# DYNAMIC MODEL LIFECYCLE MANAGER (Avoids 16GB T4 OOM)
# ==============================================================================
active_models: Dict[str, Any] = {}

def unload_models_except(keep: str):
    """Unloads all models except the one needed, freeing VRAM on T4."""
    for name in list(active_models.keys()):
        if name != keep:
            del active_models[name]
    cleanup_gpu()

# ----------------- 1. SDXL Inpainting & ControlNet -----------------
def get_sdxl_inpaint():
    if "sdxl_inpaint" not in active_models:
        unload_models_except("sdxl_inpaint")
        from diffusers import AutoPipelineForInpainting
        print("Loading SDXL Inpainting into VRAM...")
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=dtype,
            variant="fp16" if torch.cuda.is_available() else None
        ).to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        active_models["sdxl_inpaint"] = pipe
    return active_models["sdxl_inpaint"]

def get_controlnet():
    if "controlnet" not in active_models:
        unload_models_except("controlnet")
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        print("Loading ControlNet SDXL Canny into VRAM...")
        c_model = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=c_model,
            torch_dtype=dtype
        ).to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        active_models["controlnet"] = pipe
    return active_models["controlnet"]

# ----------------- 2. FLUX.2 Klein 4B -----------------
def get_flux():
    if "flux" not in active_models:
        unload_models_except("flux")
        from diffusers import DiffusionPipeline
        print("Loading FLUX.2 Klein 4B with model CPU offload...")
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        active_models["flux"] = pipe
    return active_models["flux"]

# ----------------- 3. Qwen2.5-VL 3B -----------------
def get_qwen():
    if "qwen" not in active_models:
        unload_models_except("qwen")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print("Loading Qwen2.5-VL-3B-Instruct...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=dtype,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        active_models["qwen"] = (model, processor)
    return active_models["qwen"]

# ----------------- 4. BRIA RMBG (2.0 / 1.4) -----------------
def get_rmbg():
    if "rmbg" not in active_models:
        unload_models_except("rmbg")
        from transformers import AutoModelForImageSegmentation
        print("Loading BRIA RMBG...")
        try:
            model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True).to(device)
        except Exception:
            model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device)
        model.eval()
        active_models["rmbg"] = model
    return active_models["rmbg"]

# ----------------- 5. Meta SAM 2.1 -----------------
def get_sam2():
    if "sam2" not in active_models:
        unload_models_except("sam2")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = "checkpoints/sam2.1_hiera_large.pt"
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        print("Loading Meta SAM 2.1...")
        m = build_sam2(cfg, ckpt, device=device, apply_postprocessing=True)
        active_models["sam2"] = SAM2ImagePredictor(m)
    return active_models["sam2"]

# ----------------- 6. OCR (PaddleOCR / EasyOCR) -----------------
ocr_reader = None
def get_ocr():
    global ocr_reader
    if ocr_reader is None:
        try:
            from paddleocr import PaddleOCR
            ocr_reader = ("paddle", PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available()))
        except Exception:
            import easyocr
            ocr_reader = ("easy", easyocr.Reader(['en'], gpu=torch.cuda.is_available()))
    return ocr_reader

# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models_ready": True
    }

@app.get("/api/models/status")
def models_status():
    vram = {"total_gb": 0, "allocated_gb": 0, "free_gb": 0}
    if torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc = torch.cuda.memory_allocated(0) / 1e9
        vram = {
            "device_name": torch.cuda.get_device_name(0),
            "total_gb": round(tot, 2),
            "allocated_gb": round(alloc, 2),
            "free_gb": round(tot - alloc, 2)
        }
    return {
        "connected": True,
        "vram": vram,
        "active_loaded_model": list(active_models.keys()),
        "models": {
            "sdxl_inpaint": "Loaded" if "sdxl_inpaint" in active_models else "Available",
            "controlnet": "Loaded" if "controlnet" in active_models else "Available",
            "flux_klein": "Loaded" if "flux" in active_models else "Available",
            "qwen_vl": "Loaded" if "qwen" in active_models else "Available",
            "rmbg": "Loaded" if "rmbg" in active_models else "Available",
            "sam2": "Loaded" if "sam2" in active_models else "Available",
            "ocr": "Available"
        }
    }

class InpaintReq(BaseModel):
    image_base64: str
    mask_base64: str
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, artifacts"
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    strength: Optional[float] = 0.95
    seed: Optional[int] = 42

@app.post("/inpaint")
def inpaint_endpoint(req: InpaintReq):
    try:
        img = b64_to_image(req.image_base64).convert("RGB")
        mask = b64_to_image(req.mask_base64).convert("L")
        w, h = img.size
        # Resize to multiple of 8
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        img = img.resize((w8, h8))
        mask = mask.resize((w8, h8))

        pipe = get_sdxl_inpaint()
        gen = torch.Generator(device=device).manual_seed(req.seed) if req.seed else None
        res = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=img,
            mask_image=mask,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            generator=gen
        ).images[0]
        return {"status": "success", "image_base64": image_to_b64(res.resize((w, h)))}
    except Exception as e:
        cleanup_gpu()
        raise HTTPException(status_code=500, detail=str(e))

class ControlNetReq(BaseModel):
    image_base64: str
    canny_base64: str
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality"
    conditioning_scale: Optional[float] = 0.8
    num_inference_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = 42

@app.post("/controlnet")
def controlnet_endpoint(req: ControlNetReq):
    try:
        canny = b64_to_image(req.canny_base64).convert("RGB")
        w, h = canny.size
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        canny = canny.resize((w8, h8))

        pipe = get_controlnet()
        gen = torch.Generator(device=device).manual_seed(req.seed) if req.seed else None
        res = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=canny,
            controlnet_conditioning_scale=req.conditioning_scale,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=gen
        ).images[0]
        return {"status": "success", "image_base64": image_to_b64(res.resize((w, h)))}
    except Exception as e:
        cleanup_gpu()
        raise HTTPException(status_code=500, detail=str(e))

class ImageOnlyReq(BaseModel):
    image_base64: str

@app.post("/remove-bg")
def remove_bg_endpoint(req: ImageOnlyReq):
    try:
        img = b64_to_image(req.image_base64).convert("RGB")
        orig_w, orig_h = img.size
        m = get_rmbg()
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t_img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = m(t_img)[-1].sigmoid().cpu()
        mask_np = (preds[0].squeeze().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np).resize((orig_w, orig_h))
        # Create transparent PNG
        img_rgba = img.convert("RGBA")
        img_rgba.putalpha(mask_pil)
        return {"status": "success", "image_base64": image_to_b64(img_rgba)}
    except Exception as e:
        cleanup_gpu()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr")
def ocr_endpoint(req: ImageOnlyReq):
    try:
        img = b64_to_image(req.image_base64).convert("RGB")
        engine_type, reader = get_ocr()
        extracted = []
        if engine_type == "paddle":
            res = reader.ocr(np.array(img), cls=True)
            if res and res[0]:
                for line in res[0]:
                    extracted.append({"text": line[1][0], "confidence": float(line[1][1]), "box": line[0]})
        else:
            res = reader.readtext(np.array(img))
            for box, text, conf in res:
                extracted.append({"text": text, "confidence": float(conf), "box": box})
        return {"status": "success", "elements": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_endpoint(req: ImageOnlyReq):
    try:
        img = b64_to_image(req.image_base64).convert("RGB")
        q_model, q_proc = get_qwen()
        prompt = (
            "Analyze this graphic design image in detail: critique_score (0-100), strengths, "
            "weaknesses, recommended_color_palette (hex codes), and layout suggestions in JSON format."
        )
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = q_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = q_proc(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = q_model.generate(**inputs, max_new_tokens=512)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, out_ids)]
            resp = q_proc.batch_decode(trimmed, skip_special_tokens=True)[0]
        return {"status": "success", "model": "Qwen2.5-VL", "analysis_raw": resp}
    except Exception as e:
        cleanup_gpu()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("worker_app:app", host="0.0.0.0", port=8000, reload=False)
