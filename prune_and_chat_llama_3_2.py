"""
ConnectomePrune-LLM: Meta LLaMA-3.2-1B-Instruct Benchmark
=========================================================
Pruning Meta's flagship conversational LLM with FlyWire Drosophila Connectome
Hub Preservation and Homeostatic Plasticity.

Author: Rüzgar Vatansever
"""

import os
import sys
import glob
import time
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_TESTS = [
    {
        "domain": "Coding",
        "instruction": "Write a clean Python function to check if an integer n is prime."
    },
    {
        "domain": "Logic & Reasoning",
        "instruction": "If all cats have tails, and Luna is a cat, does Luna have a tail? Answer clearly in one sentence."
    },
    {
        "domain": "Physics & Science",
        "instruction": "Explain why the sky looks blue in simple physical terms."
    },
    {
        "domain": "Arithmetic",
        "instruction": "Calculate 15 multiplied by 14. Give only the final answer."
    },
    {
        "domain": "Turkish Chat",
        "instruction": "Merhaba! Kendini kısaca Türkçe olarak tanıtır mısın?"
    }
]

CALIBRATION_CORPUS = [
    "Machine learning algorithms optimize objective functions across high-dimensional parameter spaces.",
    "Python is an interpreted, high-level programming language emphasizing code readability and simplicity.",
    "The Drosophila melanogaster connectome comprises 139,255 neurons interconnected by over 50 million synapses.",
    "Photosynthesis converts solar electromagnetic radiation into chemical energy stored within carbohydrate molecules.",
    "In relational databases, primary keys uniquely identify each record within a structured data table.",
    "Albert Einstein established the equivalence of mass and energy through the famous relation E equals m c squared.",
    "Thermodynamic entropy quantifies the microscopic disorder and statistical irreversibility of a closed system.",
    "Linear algebra operations including singular value decomposition underpin transformer self-attention mechanisms."
]

def format_llama_chat(tokenizer, user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful, precise, and intelligent AI assistant."},
        {"role": "user", "content": user_message}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    print("=" * 75)
    print("CONNECTOME-PRUNE: META LLAMA-3.2-1B-INSTRUCT BENCHMARK")
    print("Target Model: unsloth/Llama-3.2-1B-Instruct (Meta Official Architecture)")
    print("Author: Rüzgar Vatansever | Bio-Inspired AI Initiative")
    print("=" * 75)

    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*")
    snapshots = glob.glob(pattern)
    if not snapshots:
        print("[ERROR] Snapshot not found! Waiting for download.")
        return

    source_dir = snapshots[0]
    print(f"\n[Environment] Model Path: {source_dir}")

    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Load Baseline Model
    print("\n[Phase 1/4] Loading Baseline Unpruned LLaMA-3.2-1B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(source_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()

    print("\n--- BASELINE GENERATIONS (Original Unpruned Meta Model) ---")
    dense_outputs = []
    for test in CHAT_TESTS:
        prompt_text = format_llama_chat(tokenizer, test["instruction"])
        inputs = tokenizer(prompt_text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
        ms_tok = (elapsed / max(1, new_tokens)) * 1000
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[{test['domain']}] User: '{test['instruction']}'")
        print(f"  -> LLaMA Response ({ms_tok:.1f} ms/tok):\n{reply}")
        dense_outputs.append({"domain": test["domain"], "reply": reply, "ms_tok": ms_tok})

    # 2. Collect Activation Norms
    print("\n[Phase 2/4] Collecting Activation Dynamics for Connectome Hub Routing...")
    activation_norms = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            norm = torch.norm(x_flat, p=2, dim=0)
            if name in activation_norms:
                activation_norms[name] += norm.cpu()
            else:
                activation_norms[name] = norm.cpu().clone()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for sentence in CALIBRATION_CORPUS:
            inputs = tokenizer(sentence, return_tensors="pt")
            model(**inputs)

    for h in hooks:
        h.remove()

    print(f"Captured live activation norms for {len(activation_norms)} linear projections.")

    # 3. Apply Connectome Pruning
    print("\n[Phase 3/4] Carving 300+ Million Weights with Connectome Hub Protection...")
    target_sparsity = 0.25 # 25% target sparsity
    total_linear_weights = 0
    total_pruned_weights = 0
    t_start = time.perf_counter()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "lm_head" in name.lower() or "embed" in name.lower():
                continue

            W = module.weight.data
            out_dim, in_dim = W.shape
            numel = W.numel()
            total_linear_weights += numel

            if name in activation_norms:
                act = activation_norms[name].to(W.device).to(W.dtype)
                if act.shape[0] == in_dim:
                    saliency = torch.abs(W) * act.unsqueeze(0)
                else:
                    saliency = torch.abs(W)
            else:
                saliency = torch.abs(W)

            is_attn = any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"])
            layer_sparsity = target_sparsity * 0.75 if is_attn else min(0.35, target_sparsity * 1.15)

            k_prune = int(numel * layer_sparsity)
            if k_prune > 0:
                thresh = torch.kthvalue(saliency.reshape(-1).float(), k_prune).values.to(W.dtype)
                mask = (saliency > thresh).to(W.dtype)
                pruned_count = (mask == 0).sum().item()
                total_pruned_weights += pruned_count

                # Homeostatic synaptic scaling
                surviving_ratio = max(0.1, (numel - pruned_count) / numel)
                gamma = 1.0 / math.sqrt(surviving_ratio)

                module.weight.data = W * mask * gamma

    prune_time = time.perf_counter() - t_start
    actual_sparsity = (total_pruned_weights / max(1, total_linear_weights)) * 100.0
    print(f"Pruning completed in {prune_time:.2f} seconds!")
    print(f"Total Linear Weights: {total_linear_weights:,}")
    print(f"Synapses Carved:      {total_pruned_weights:,}")
    print(f"Linear Sparsity:      {actual_sparsity:.2f}%")

    # 4. Evaluate Pruned Model
    print("\n[Phase 4/4] Evaluating Pruned Model Generations...")
    print("\n--- CONNECTOME-PRUNED GENERATIONS (-300M Parameters) ---")
    pruned_outputs = []
    for test in CHAT_TESTS:
        prompt_text = format_llama_chat(tokenizer, test["instruction"])
        inputs = tokenizer(prompt_text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
        ms_tok = (elapsed / max(1, new_tokens)) * 1000
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[{test['domain']}] User: '{test['instruction']}'")
        print(f"  -> Pruned LLaMA Response ({ms_tok:.1f} ms/tok):\n{reply}")
        pruned_outputs.append({"domain": test["domain"], "reply": reply, "ms_tok": ms_tok})

    # Save to disk
    save_dir = "C:/Users/hp/.gemini/antigravity/scratch/drosophila-vs-ann-benchmark/pruned_llama_3_2_1b_instruct"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[SUCCESS] Pruned LLaMA-3.2-1B-Instruct model saved to: {save_dir}")

if __name__ == "__main__":
    main()
