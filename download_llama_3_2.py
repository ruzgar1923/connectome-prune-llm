import time
from huggingface_hub import snapshot_download

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
print(f"Starting download for Meta's {MODEL_ID} (1.23 Billion Parameters)...")
t0 = time.time()

path = snapshot_download(
    repo_id=MODEL_ID,
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
    resume_download=True
)

elapsed = time.time() - t0
print(f"Downloaded successfully in {elapsed:.2f} seconds to: {path}")
