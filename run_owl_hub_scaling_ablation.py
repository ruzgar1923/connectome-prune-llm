"""
OWL + Hub + Scaling Ablation Experiment (Incremental Auto-Save)
==============================================================
Evaluates:
1. Standard OWL (Yin et al., 2024: depth-dependent layer sparsity)
2. OWL + Scaling Only
3. OWL + Hub Protection Only (OWL inter-layer + Attention protective intra-layer)
4. OWL + Hub + Scaling (Unified framework)
5. ConnectomePrune (Iso-Sparsity Attention Protection + Output Scaling)

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

def get_model_path():
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*")
    snaps = glob.glob(pattern)
    return snaps[0] if snaps else None

def main():
    print("=" * 75)
    print("RUNNING OWL + HUB + SCALING MULTI-COMBINATION ABLATION")
    print("=" * 75)

    model_dir = get_model_path()
    if not model_dir:
        print("[ERROR] Model snapshot not found!")
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
        "In artificial intelligence, post-training pruning eliminates redundant parameters without catastrophic forgetting.",
        "Electromagnetism governs the interactions between charged particles via electromagnetic field equations.",
        "Computer architectures utilize multi-level caching to hide main memory latency and maximize compute utilization.",
        "Spiking neural networks leverage temporal event-driven activation sparsity to minimize dynamic energy draw.",
        "Algorithmic complexity theory categorizes computational problems according to their asymptotic resource requirements."
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

    base_target = 0.2710
    num_layers = 16
    owl_layer_sparsity = {}
    for l in range(num_layers):
        factor = 1.0 + 0.35 * math.sin(math.pi * l / (num_layers - 1)) - 0.15
        owl_layer_sparsity[l] = base_target * factor

    mean_sp = sum(owl_layer_sparsity.values()) / num_layers
    norm_factor = base_target / mean_sp
    for l in range(num_layers):
        owl_layer_sparsity[l] = round(owl_layer_sparsity[l] * norm_factor, 4)

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "owl_hub_scaling_ablation.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    configs = [
        {"name": "Standard OWL (2024)", "hub_prot": False, "scaling": False},
        {"name": "OWL + Scaling Only", "hub_prot": False, "scaling": True},
        {"name": "OWL + Hub Protection Only", "hub_prot": True, "scaling": False},
        {"name": "OWL + Hub + Scaling (Unified)", "hub_prot": True, "scaling": True},
        {"name": "ConnectomePrune (Ours: Iso-Sparsity + Scaling)", "hub_prot": True, "scaling": True, "custom_iso": True}
    ]

    results = []

    for cfg in configs:
        print(f"\n[Evaluating]: {cfg['name']} ...")
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)

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
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name.lower() and "embed" not in name.lower():
                W = module.weight.data
                numel = W.numel()
                is_attn = any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"])
                
                layer_idx = 0
                for part in name.split("."):
                    if part.isdigit():
                        layer_idx = int(part)
                        break

                if cfg.get("custom_iso", False):
                    sp = 0.1875 if is_attn else 0.2875
                else:
                    base_l_sp = owl_layer_sparsity.get(layer_idx, 0.2710)
                    if cfg["hub_prot"]:
                        sp = base_l_sp * 0.75 if is_attn else base_l_sp * 1.05
                    else:
                        sp = base_l_sp

                act = act_norms.get(name, torch.ones(W.shape[1])).to(W.device).to(W.dtype)
                saliency = torch.abs(W) * act.unsqueeze(0)
                k_prune = int(numel * sp)
                if k_prune > 0:
                    kth = torch.kthvalue(saliency.reshape(-1).float(), k_prune).values.to(W.dtype)
                    mask = (saliency > kth).to(W.dtype)
                    p_cnt = (mask == 0).sum().item()
                    is_output = any(k in name.lower() for k in ["o_proj", "down_proj"])
                    gamma = 1.0 / math.sqrt(max(0.05, (numel - p_cnt)/numel)) if (cfg["scaling"] and is_output) else 1.0
                    module.weight.data = W * mask * gamma
                    total_pruned += p_cnt
                    total_linear += numel

        actual_sp = (total_pruned / max(1, total_linear)) * 100.0
        ppl = eval_ppl(model)
        elapsed = time.perf_counter() - t0

        item = {
            "name": cfg["name"],
            "actual_sparsity_pct": round(actual_sp, 2),
            "ppl_512": round(ppl, 2),
            "elapsed_seconds": round(elapsed, 1)
        }
        results.append(item)
        print(f"  -> Sparsity: {actual_sp:.2f}% | PPL: {ppl:.2f} in {elapsed:.1f}s")
        
        # Save incrementally
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        del model
        gc.collect()

    print("\n" + "=" * 75)
    print("ALL ABLATIONS COMPLETED! Results saved to:", out_file)
    print("=" * 75)

if __name__ == "__main__":
    main()
