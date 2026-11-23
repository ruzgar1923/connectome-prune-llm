import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connectome_prune_llm.sculptor import ConnectomeSculptor

pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*")
source_dir = glob.glob(pattern)[0]
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sculpted_qwen_1.5b")

print("Re-sculpting Qwen 2.5-1.5B with protected vocabulary embeddings...")
sculptor = ConnectomeSculptor(target_sparsity=0.25, hub_protection_ratio=0.20, homeostatic_scaling=True)
report = sculptor.sculpt_model_directory(source_dir, target_dir)
print("Re-sculpt complete! Report:", report)
