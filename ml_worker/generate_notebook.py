import json

with open("ml_worker/worker_app.py", "r", encoding="utf-8") as f:
    code_script = f.read()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# VariantForge AI — Dedicated Google Colab T4 GPU Remote Inference Server\n",
            "### Turnkey inference engine for all 7 models with on-demand VRAM management\n",
            "1. Set Runtime: **Runtime -> Change runtime type -> T4 GPU**\n",
            "2. Under Secrets (key icon on left), add `NGROK_AUTHTOKEN` (and optionally `HF_TOKEN`)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Verify GPU & VRAM\n",
            "import torch\n",
            "print('CUDA Available:', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU Device:', torch.cuda.get_device_name(0))\n",
            "    print('Total VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install Dependencies\n",
            "!pip install -q --upgrade pip\n",
            "!pip install -q diffusers transformers accelerate safetensors torchvision\n",
            "!pip install -q opencv-python pillow numpy fastapi uvicorn pyngrok easyocr\n",
            "!git clone -q https://github.com/facebookresearch/sam2.git || true\n",
            "!pip install -q \"git+https://github.com/facebookresearch/sam2.git\" || true\n",
            "!mkdir -p checkpoints\n",
            "!wget -q -nc -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt || true\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Write FastAPI Worker Application\n",
            f"code = {repr(code_script)}\n",
            "with open('worker_app.py', 'w', encoding='utf-8') as f:\n",
            "    f.write(code)\n",
            "print('worker_app.py written successfully!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Connect ngrok & Start Server\n",
            "from pyngrok import ngrok\n",
            "import uvicorn\n",
            "try:\n",
            "    from google.colab import userdata\n",
            "    token = userdata.get('NGROK_AUTHTOKEN')\n",
            "    if token:\n",
            "        ngrok.set_auth_token(token)\n",
            "except Exception as e:\n",
            "    print('Note on secrets:', e)\n",
            "\n",
            "ngrok.kill()\n",
            "tunnel = ngrok.connect(8000)\n",
            "print('=' * 75)\n",
            "print('🔥 VARIANTFORGE AI T4 GPU WORKER ONLINE!')\n",
            "print('👉 COPY THIS PUBLIC URL INTO YOUR LOCAL APP:', tunnel.public_url)\n",
            "print('=' * 75)\n",
            "\n",
            "from worker_app import app\n",
            "uvicorn.run(app, host='0.0.0.0', port=8000)\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {"colab": {"name": "VariantForge_GPU_Inference.ipynb"}},
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("ml_worker/VariantForge_GPU_Inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook generated successfully!")
