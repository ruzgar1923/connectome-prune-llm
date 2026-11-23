import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B"
print(f"Starting download for {MODEL_ID} (1.54 Billion Parameters)...")
t0 = time.time()

path = snapshot_download(
    repo_id=MODEL_ID,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    resume_download=True
)

elapsed = time.time() - t0
print(f"Downloaded successfully in {elapsed:.2f} seconds to: {path}")
