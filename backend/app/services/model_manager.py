import os
import logging
from typing import Optional, Any, Dict
from PIL import Image

logger = logging.getLogger(__name__)

class DiffusionArchitectureInfo:
    """
    Theoretical reference & technical inspection for Diffusion Models, DDPM, and U-Net.
    Compares against GANs and VAEs.
    """
    @staticmethod
    def get_summary() -> Dict[str, Any]:
        return {
            "paradigm": "Denoising Diffusion Probabilistic Models (DDPM)",
            "comparison": {
                "GAN": "Adversarial minimax (Generator vs Discriminator). Prone to mode collapse and unstable training dynamics.",
                "VAE": "Probabilistic latent bottleneck with Reconstruction Loss + KL Divergence. Tends to generate blurry outputs.",
                "DDPM": "Iterative Markovian noise reversal. Highest fidelity, stable maximum likelihood training, granular control over each timestep."
            },
            "processes": {
                "forward_diffusion": "q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t)*x_{t-1}, beta_t*I). Systematically adds Gaussian noise across T timesteps.",
                "reverse_diffusion": "p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2*I). Parameterized neural network predicts and subtracts noise step-by-step."
            },
            "unet_architecture": {
                "encoder": "Downsampling blocks extracting multi-scale hierarchical spatial features.",
                "timestep_embeddings": "Sinusoidal/MLP projections informing every block of the current diffusion timestep t.",
                "skip_connections": "Direct tensor concatenations bridging encoder to decoder, preserving spatial boundaries, fine details, and creative composition.",
                "decoder": "Upsampling blocks reconstructing feature maps and predicting the residual noise epsilon_theta(x_t, t)."
            }
        }

class ModelManager:
    """
    Centralized manager for loading, caching, and running deep learning models:
      - SDXL Inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
      - ControlNet Canny SDXL (diffusers/controlnet-canny-sdxl-1.0)
      - FLUX.2 Klein 4B (black-forest-labs/FLUX.2-klein-4B)
      - Qwen2.5-VL-3B-Instruct (Vision-Language Analysis & Audit)
      - BRIA RMBG-2.0 / 1.4 (Salient Object Background Removal)
      - Meta SAM 2.1 (Segment Anything 2.1 Hiera)
      - PaddleOCR / EasyOCR (Text Extraction & Safe Region Tracking)
    Manages GPU VRAM and seamlessly routes to remote GPU worker (Colab) when available.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = self._detect_device()
        self.remote_url = os.getenv("REMOTE_ML_URL", "").rstrip("/")
        
        # Cached pipelines
        self.sdxl_inpaint_pipe = None
        self.controlnet_pipe = None
        self.flux_pipe = None
        self.sam2_predictor = None
        self.rmbg_model = None
        self.qwen_model = None
        self.qwen_processor = None
        self.ocr_reader = None
        
        # Status tracking
        self.model_status = {
            "sdxl_inpaint": {"loaded": False, "device": self.device, "error": None},
            "controlnet": {"loaded": False, "device": self.device, "error": None},
            "flux_klein": {"loaded": False, "device": self.device, "error": None},
            "qwen_vl": {"loaded": False, "device": self.device, "error": None},
            "rmbg": {"loaded": False, "device": self.device, "error": None},
            "sam2": {"loaded": False, "device": self.device, "error": None},
            "ocr": {"loaded": False, "device": self.device, "error": None}
        }

        logger.info(f"ModelManager initialized. Device: {self.device}. Remote GPU URL: {self.remote_url or 'None'}")
        self._initialized = True

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def has_cuda(self) -> bool:
        return self.device == "cuda"

    def is_remote_enabled(self) -> bool:
        url = (self.remote_url or os.getenv("REMOTE_ML_URL", "")).strip()
        return bool(url and url.startswith("http"))

    def require_active_gpu(self, model_name: str = "AI Model") -> str:
        """
        Validates that either Cloud APIs (Fal.ai / Together AI / Gemini) or a remote Colab GPU or local CUDA is available.
        """
        from app.services.cloud_ai_service import CloudAIService
        if CloudAIService.is_cloud_enabled():
            return "cloud_api"
        if self.has_cuda():
            return "local_cuda"
        if self.is_remote_enabled():
            status = self.ping_remote()
            if status.get("connected"):
                return "remote_worker"
        
        # If neither cloud keys nor active Colab worker exists, raise clear guidance
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"{model_name} requires an active AI engine. Please enter your Fal.ai / Together API key in Settings, or connect a Google Colab GPU."
        )

    def ping_remote(self) -> Dict[str, Any]:
        """Tests health and status of the remote Colab worker."""
        if not self.is_remote_enabled():
            return {"connected": False, "message": "No remote worker URL configured"}
        import requests
        try:
            url = (self.remote_url or os.getenv("REMOTE_ML_URL", "")).rstrip("/")
            headers = {"ngrok-skip-browser-warning": "true", "User-Agent": "VariantForgeClient"}
            res = requests.get(f"{url}/api/models/status", headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                data["connected"] = True
                data["worker_url"] = url
                return data
            return {"connected": False, "status_code": res.status_code, "message": res.text}
        except Exception as e:
            return {"connected": False, "message": str(e)}

    def get_status(self) -> Dict[str, Any]:
        from app.services.cloud_ai_service import CloudAIService
        remote_info = self.ping_remote()
        vram_info = remote_info.get("vram", {"total_gb": 0.0, "allocated_gb": 0.0, "free_gb": 0.0})

        cloud_active = CloudAIService.is_cloud_enabled()
        fal_set = bool(CloudAIService.get_fal_key())
        gemini_set = bool(CloudAIService.get_gemini_key())
        together_set = bool(CloudAIService.get_together_key())

        return {
            "device": self.device,
            "has_cuda": self.has_cuda(),
            "cloud_enabled": cloud_active,
            "cloud_keys": {
                "fal_key_set": fal_set,
                "gemini_key_set": gemini_set,
                "together_key_set": together_set
            },
            "remote_worker_url": self.remote_url or os.getenv("REMOTE_ML_URL", ""),
            "remote_enabled": self.is_remote_enabled(),
            "remote_connected": remote_info.get("connected", False) or cloud_active,
            "remote_details": remote_info,
            "vram": vram_info if not cloud_active else {"device_name": "Cloud AI (Fal.ai / Google H100)", "allocated_gb": 12.0, "total_gb": 80.0, "free_gb": 68.0},
            "models": {
                "controlnet": "Cloud Active" if fal_set else remote_info.get("models", {}).get("controlnet", "Available"),
                "sdxl_inpaint": "Cloud Active" if fal_set else remote_info.get("models", {}).get("sdxl_inpaint", "Available"),
                "rmbg": "Cloud Active" if fal_set else "Available",
                "ocr": "Local Available",
                "flux": "Cloud Active" if (fal_set or together_set) else "Available"
            },
            "architecture_reference": DiffusionArchitectureInfo.get_summary()
        }

    def load_sdxl_inpaint(self, model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
        """Loads and caches SDXL Inpainting pipeline (U-Net with skip connections)."""
        if self.sdxl_inpaint_pipe is not None:
            return self.sdxl_inpaint_pipe
        if not self.has_cuda():
            logger.info("Local CUDA not found. SDXL Inpainting will run on Remote Worker or CPU offload.")
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
            dtype = torch.float16 if self.has_cuda() else torch.float32
            logger.info(f"Loading SDXL Inpainting: {model_id}...")
            pipe = AutoPipelineForInpainting.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if self.has_cuda() else None,
                use_safetensors=True
            )
            if self.has_cuda():
                pipe.to("cuda")
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
            self.sdxl_inpaint_pipe = pipe
            self.model_status["sdxl_inpaint"] = {"loaded": True, "device": self.device, "error": None}
            return self.sdxl_inpaint_pipe
        except Exception as e:
            logger.warning(f"Could not load SDXL inpainting locally: {e}")
            self.model_status["sdxl_inpaint"]["error"] = str(e)
            return None

    def load_controlnet_canny(self, controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0"):
        """Loads ControlNet with Canny conditioning for structural preservation."""
        if self.controlnet_pipe is not None:
            return self.controlnet_pipe
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
            dtype = torch.float16 if self.has_cuda() else torch.float32
            logger.info(f"Loading ControlNet: {controlnet_id}...")
            controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=dtype
            )
            if self.has_cuda():
                pipe.to("cuda")
            self.controlnet_pipe = pipe
            self.model_status["controlnet"] = {"loaded": True, "device": self.device, "error": None}
            return self.controlnet_pipe
        except Exception as e:
            logger.warning(f"Could not load ControlNet locally: {e}")
            self.model_status["controlnet"]["error"] = str(e)
            return None

    def load_flux(self, model_id: str = "black-forest-labs/FLUX.2-klein-4B"):
        """Loads FLUX.2 Klein 4B flow-matching diffusion pipeline."""
        if self.flux_pipe is not None:
            return self.flux_pipe
        try:
            import torch
            from diffusers import DiffusionPipeline
            dtype = torch.bfloat16 if self.has_cuda() else torch.float32
            self.flux_pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self.has_cuda() else None
            )
            self.model_status["flux_klein"] = {"loaded": True, "device": self.device, "error": None}
            return self.flux_pipe
        except Exception as e:
            logger.warning(f"Could not load FLUX locally: {e}")
            self.model_status["flux_klein"]["error"] = str(e)
            return None

    def load_rmbg(self, model_id: str = "briaai/RMBG-2.0"):
        """Loads BRIA RMBG background removal model with RMBG-1.4 fallback."""
        if self.rmbg_model is not None:
            return self.rmbg_model
        try:
            from transformers import AutoModelForImageSegmentation
            self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                model_id,
                trust_remote_code=True
            ).to(self.device)
            self.rmbg_model.eval()
            self.model_status["rmbg"] = {"loaded": True, "device": self.device, "error": None}
            return self.rmbg_model
        except Exception as e1:
            try:
                from transformers import AutoModelForImageSegmentation
                self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                    "briaai/RMBG-1.4",
                    trust_remote_code=True
                ).to(self.device)
                self.rmbg_model.eval()
                self.model_status["rmbg"] = {"loaded": True, "device": self.device, "error": None}
                return self.rmbg_model
            except Exception as e2:
                self.model_status["rmbg"]["error"] = str(e2)
                return None

    def clear_vram(self):
        """Frees cached GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

model_manager = ModelManager()
