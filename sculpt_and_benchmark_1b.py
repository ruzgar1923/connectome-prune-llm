"""
Connectome Sculptor Benchmark on Billion-Scale Model: Qwen 2.5-1.5B
===================================================================
Author: Rüzgar Vatansever
"""

import os
import sys
import time
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connectome_prune_llm.sculptor import ConnectomeSculptor
from connectome_prune_llm.evaluator import ModelEvaluator

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "Artificial intelligence can be defined as",
    "In physics, Albert Einstein is famous for the theory of"
]

def main():
    print("=" * 75)
    print("CONNECTOME SCULPTOR 1.5 BILLION PARAMETER BENCHMARK")
    print("Model: Qwen/Qwen2.5-1.5B | Author: Rüzgar Vatansever")
    print("=" * 75)

    # 1. Locate downloaded snapshot
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*")
    snapshots = glob.glob(pattern)
    if not snapshots:
        print("[ERROR] Snapshot not found! Please ensure download has completed.")
        return

    source_dir = snapshots[0]
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sculpted_qwen_1.5b")
    print(f"[Source Snapshot]: {source_dir}")
    print(f"[Target Pruned]:   {target_dir}\n")

    # 2. Evaluate Baseline Unpruned Model
    print("[Phase 1/3] Loading Baseline Dense Model for Evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True)
    dense_model = AutoModelForCausalLM.from_pretrained(source_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    evaluator_dense = ModelEvaluator(dense_model, tokenizer, device="cpu")

    print("[Baseline Output Generation]:")
    dense_results = evaluator_dense.generate_responses(PROMPTS, max_tokens=25)
    for r in dense_results:
        print(f"  * \"{r['completion']}\" ({r['ms_per_token']:.1f} ms/tok)")

    # Free dense model from RAM before sculpting
    del dense_model
    import gc
    gc.collect()

    # 3. Sculpt the 1.5B Model Shards
    print("\n[Phase 2/3] Activating ConnectomeSculptor Sharded Carving...")
    sculptor = ConnectomeSculptor(
        target_sparsity=0.25,
        hub_protection_ratio=0.20,
        homeostatic_scaling=True
    )
    t0 = time.perf_counter()
    report = sculptor.sculpt_model_directory(source_dir, target_dir)
    sculpt_elapsed = time.perf_counter() - t0

    # 4. Evaluate Pruned Sculpted Model
    print("\n[Phase 3/3] Evaluating Sculpted Pruned Model (0 RAM-exhaustion)...")
    pruned_model = AutoModelForCausalLM.from_pretrained(target_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    evaluator_pruned = ModelEvaluator(pruned_model, tokenizer, device="cpu")

    print("[Sculpted Pruned Model Output Generation]:")
    pruned_results = evaluator_pruned.generate_responses(PROMPTS, max_tokens=25)
    for r in pruned_results:
        print(f"  * \"{r['completion']}\" ({r['ms_per_token']:.1f} ms/tok)")

    # 5. Compile Final Benchmark Report
    summary = {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "author": "Rüzgar Vatansever",
        "total_parameters": report["total_params"],
        "pruned_synapses": report["pruned_params"],
        "net_sparsity_pct": report["net_sparsity"],
        "sculpt_time_seconds": sculpt_elapsed,
        "dense_generations": dense_results,
        "pruned_generations": pruned_results
    }

    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "qwen_1.5b_sculptor_benchmark.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("1.5 BILLION PARAMETER BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total Parameters: {report['total_params']:,}")
    print(f"Synapses Carved:  {report['pruned_params']:,} ({report['net_sparsity']:.2f}% REMOVED)")
    print(f"Full Report Saved: {report_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
