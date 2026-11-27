"""
Qwen2.5-1.5B Multi-Model Benchmark Engine
==========================================
Validates ConnectomePrune generalization beyond LLaMA architecture on Qwen2.5-1.5B (28 layers, 1.54B params).
Evaluates:
- Dense Baseline
- Standard Wanda (25.0% uniform)
- ConnectomePrune (Hub Protected: 18.0% Attn, 25.9% MLP)
- Inverse-Hub Control (35.0% Attn, 23.7% MLP)
- High-Sparsity ConnectomePrune (40.0%)

Author: Rüzgar Vatansever
"""

import os
import sys
import gc
import json
import math
import time
import glob
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_qwen_path():
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*")
    snaps = glob.glob(pattern)
    return snaps[0] if snaps else None

def main():
    print("=" * 70)
    print("MULTI-MODEL EVALUATION: Qwen2.5-1.5B (28 Layers, 1.54B Parameters)")
    print("=" * 70)

    model_dir = get_qwen_path()
    if not model_dir:
        print("[ERROR] Qwen2.5-1.5B snapshot not found!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    calibration_corpus = [
        "In artificial intelligence, post-training pruning compresses parameter count while preserving representations.",
        "Differential geometry provides the foundational framework for Einstein's general theory of relativity.",
        "Deep convolutional architectures learn hierarchical spatial representations from raw sensor data.",
        "Transformer multi-head self-attention mechanisms establish dynamic token routing across sequences.",
        "Modern neuromorphic hardware leverages event-driven spike accumulation to achieve sub-watt operation."
    ]

    eval_corpus = [
        "Electromagnetism governs the interactions between charged particles via Maxwell's field equations.",
        "Computer architectures utilize multi-level caching to hide main memory latency and maximize throughput.",
        "Spiking neural networks leverage temporal activation sparsity to minimize dynamic power dissipation.",
        "Algorithmic complexity theory categorizes problems according to their asymptotic resource bounds."
    ]

    def eval_ppl(m):
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in eval_corpus:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                ids = enc["input_ids"]
                if ids.shape[1] < 4: continue
                out = m(ids, labels=ids.clone())
                total_loss += out.loss.item() * ids.shape[1]
                total_tokens += ids.shape[1]
        return math.exp(total_loss / max(1, total_tokens))

    configs = [
        {"name": "Dense Baseline (Qwen 1.5B)", "attn_sp": 0.0, "mlp_sp": 0.0, "scaling": False},
        {"name": "Standard Wanda (25.0%)", "attn_sp": 0.250, "mlp_sp": 0.250, "scaling": False},
        {"name": "ConnectomePrune (25.0%)", "attn_sp": 0.180, "mlp_sp": 0.259, "scaling": True},
        {"name": "Inverse-Hub Control (25.0%)", "attn_sp": 0.350, "mlp_sp": 0.237, "scaling": True},
        {"name": "ConnectomePrune (40.0%)", "attn_sp": 0.280, "mlp_sp": 0.416, "scaling": True}
    ]

    results = []

    for cfg in configs:
        print(f"\n[Evaluating]: {cfg['name']} ...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)

        if cfg["attn_sp"] > 0 or cfg["mlp_sp"] > 0:
            # Collect norms
            act_norms = {}
            hooks = []
            def make_h(name):
                def fn(mod, inp, out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    norm = torch.norm(x, p=2, dim=0).cpu()
                    act_norms[name] = act_norms.get(name, 0) + norm
                return fn
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear) and "lm_head" not in n.lower() and "embed" not in n.lower():
                    hooks.append(m.register_forward_hook(make_h(n)))
            with torch.no_grad():
                for t in calibration_corpus: model(**tokenizer(t, return_tensors="pt"))
            for h in hooks: h.remove()

            # Prune
            total_pruned, total_linear = 0, 0
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear) and "lm_head" not in n.lower() and "embed" not in n.lower():
                    W = m.weight.data
                    numel = W.numel()
                    is_attn = any(k in n.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"])
                    sp = cfg["attn_sp"] if is_attn else cfg["mlp_sp"]
                    act = act_norms.get(n, torch.ones(W.shape[1])).to(W.device).to(W.dtype)
                    saliency = torch.abs(W) * act.unsqueeze(0)
                    k_prune = int(numel * sp)
                    if k_prune > 0:
                        kth = torch.kthvalue(saliency.reshape(-1).float(), k_prune).values.to(W.dtype)
                        mask = (saliency > kth).to(W.dtype)
                        p_cnt = (mask == 0).sum().item()
                        is_output = any(k in n.lower() for k in ["o_proj", "down_proj"])
                        gamma = 1.0 / math.sqrt(max(0.05, (numel - p_cnt)/numel)) if (cfg["scaling"] and is_output) else 1.0
                        m.weight.data = W * mask * gamma
                        total_pruned += p_cnt
                        total_linear += numel
            actual_sp = (total_pruned / max(1, total_linear)) * 100.0
        else:
            actual_sp = 0.0

        ppl = eval_ppl(model)
        elapsed = time.perf_counter() - t0
        item = {
            "name": cfg["name"],
            "actual_sparsity_pct": actual_sp,
            "ppl_512": ppl,
            "elapsed_seconds": round(elapsed, 1)
        }
        results.append(item)
        print(f"  -> Sparsity: {actual_sp:.2f}% | PPL: {ppl:.2f} in {elapsed:.1f}s")
        del model
        gc.collect()

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "qwen_q1_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Qwen benchmark completed! Saved to: {out_file}")

if __name__ == "__main__":
    main()
