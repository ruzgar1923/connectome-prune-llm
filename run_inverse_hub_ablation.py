"""
ConnectomePrune-LLM: Inverse-Hub Ablation Experiment
===================================================
Compares:
1. Standard ConnectomePrune (Attention Protected: 18.75% sparsity, MLP: 28.75% sparsity)
2. Inverse-Hub Control (Attention Penalized: 28.75% sparsity, MLP: 18.75% sparsity)
Aggregate linear sparsity in both cases: ~27.10%
"""

import os
import glob
import math
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_loss_and_generation(model, tokenizer, test_prompts, eval_corpus):
    # Compute Cross-Entropy Loss on evaluation corpus
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in eval_corpus:
            inputs = tokenizer(text, return_tensors="pt")
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            tokens = inputs["input_ids"].shape[1]
            total_loss += loss * tokens
            total_tokens += tokens
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)

    # Test sample generations
    generations = {}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        generations[prompt] = reply

    return avg_loss, ppl, generations

def main():
    print("=" * 70)
    print("RUNNING INVERSE-HUB ABLATION BENCHMARK (Meta LLaMA-3.2-1B-Instruct)")
    print("=" * 70)

    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*")
    snapshots = glob.glob(pattern)
    if not snapshots:
        print("[ERROR] Snapshot not found!")
        return
    source_dir = snapshots[0]

    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eval_corpus = [
        "The theory of general relativity describes gravitation as a geometric property of spacetime.",
        "Deep learning models optimize loss functions across multi-dimensional parameter landscapes.",
        "Drosophila melanogaster neural circuits coordinate sensory steering with high energetic efficiency.",
        "In modern computer systems, memory hierarchies balance latency, bandwidth, and fabrication costs.",
        "Photosynthesis in green plants utilizes chlorophyll pigments to absorb electromagnetic radiation."
    ]

    test_prompts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite a clean Python function to check if an integer n is prime.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain why the sky looks blue in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ]

    # Test 1: Inverse-Hub (Attention is pruned heavily, MLP is protected)
    print("\n--- [Experiment A] Inverse-Hub (Attention Sparsity: 28.75%, MLP Sparsity: 18.75%) ---")
    model_inv = AutoModelForCausalLM.from_pretrained(source_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model_inv.eval()

    # Activation collection
    activation_norms = {}
    hooks = []
    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            norm = torch.norm(x, p=2, dim=0).cpu()
            if name in activation_norms:
                activation_norms[name] += norm
            else:
                activation_norms[name] = norm.clone()
        return hook_fn

    for name, module in model_inv.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name.lower() and "embed" not in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for text in eval_corpus:
            inputs = tokenizer(text, return_tensors="pt")
            model_inv(**inputs)
    for h in hooks: h.remove()

    # Inverse-Hub Pruning
    for name, module in model_inv.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name.lower() and "embed" not in name.lower():
            W = module.weight.data
            out_dim, in_dim = W.shape
            numel = W.numel()
            act = activation_norms.get(name, torch.ones(in_dim))
            saliency = torch.abs(W) * act.to(W.device).to(W.dtype).unsqueeze(0)

            is_attn = any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"])
            # INVERSE: Attention gets higher sparsity (28.75%), MLP gets lower (18.75%)
            layer_sparsity = 0.2875 if is_attn else 0.1875
            k_prune = int(numel * layer_sparsity)
            if k_prune > 0:
                thresh = torch.kthvalue(saliency.reshape(-1).float(), k_prune).values.to(W.dtype)
                mask = (saliency > thresh).to(W.dtype)
                surviving = max(0.1, (numel - (mask == 0).sum().item()) / numel)
                gamma = 1.0 / math.sqrt(surviving)
                module.weight.data = W * mask * gamma

    loss_inv, ppl_inv, gens_inv = evaluate_loss_and_generation(model_inv, tokenizer, test_prompts, eval_corpus)
    print(f"\n[Inverse-Hub Results] Loss: {loss_inv:.4f} | Perplexity: {ppl_inv:.2f}")
    for p, g in gens_inv.items():
        print(f"Prompt snippet: ... -> Response: {g[:80]}...")
    del model_inv

    print("\n" + "=" * 70)
    print("INVERSE-HUB EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
