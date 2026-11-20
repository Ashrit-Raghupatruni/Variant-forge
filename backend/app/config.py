import os
from pathlib import Path
from pydantic import BaseModel

# Automatically load .env file if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

class Settings(BaseModel):
    PROJECT_NAME: str = "VariantForge AI"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days
    
    # Upload storage directory
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
    
    # ML Provider settings: "mock", "local", or "remote" (e.g., Colab / GPU worker)
    ML_PROVIDER: str = os.getenv("ML_PROVIDER", "mock")
    REMOTE_ML_URL: str = os.getenv("REMOTE_ML_URL", "")
    
    # FLUX Settings
    FLUX_MODEL_ID: str = "black-forest-labs/FLUX.2-klein-4B"
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
