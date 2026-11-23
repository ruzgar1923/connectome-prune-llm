"""
Calibrated Connectome-Pruning with Homeostatic Plasticity
=========================================================
Implements activation-aware saliency (Wanda/SparseGPT principle) combined with
Drosophila connectome Hub protection and biological homeostatic synaptic scaling.

Author: Rüzgar Vatansever
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

# Standard diverse calibration text (mimics sensory exposure in biological eclosion)
CALIBRATION_CORPUS = [
    "The solar system consists of the Sun and the objects that orbit around it.",
    "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
    "Albert Einstein developed the theory of relativity, which revolutionized theoretical physics.",
    "In computer science, algorithms are step-by-step procedures used for calculation and data processing.",
    "The human heart pumps oxygenated blood through the circulatory system to nourish all body cells.",
    "Drosophila melanogaster, commonly known as the fruit fly, is a primary model organism in genetics and neurobiology.",
    "Machine learning models discover patterns within high-dimensional numerical datasets.",
    "Water exists in liquid, solid, and gaseous states depending on ambient temperature and pressure."
]

def collect_activation_norms(model, tokenizer, calibration_texts):
    """
    Hooks into linear layers during forward passes over calibration text
    to calculate true empirical input activation norms ||X_j||_2.
    """
    print("[Calibration] Registering neural activity sensors (forward hooks)...")
    activation_norms = {}
    hooks = []
    
    def get_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach() # [Batch, Seq, In_Dim]
            x_flat = x.reshape(-1, x.shape[-1]) # [Total_Tokens, In_Dim]
            norm = torch.norm(x_flat, p=2, dim=0) # [In_Dim]
            if name in activation_norms:
                activation_norms[name] += norm
            else:
                activation_norms[name] = norm.clone()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))
            
    # Run calibration texts through model to observe live neural traffic
    model.eval()
    with torch.no_grad():
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
            
    # Remove hooks
    for h in hooks:
        h.remove()
        
    print(f"[Calibration] Successfully captured live activation dynamics for {len(activation_norms)} linear layers.")
    return activation_norms

def apply_homeostatic_connectome_prune(model, activation_norms, target_sparsity=0.25):
    """
    Applies Connectome-Pruning with:
    1. Saliency = |W_ij| * ||X_j||_2 (protecting outlier communication channels)
    2. Attention Head Hub Protection (Central Complex principle)
    3. Homeostatic Synaptic Scaling (LTP compensation to prevent voltage collapse)
    """
    print(f"\n[ConnectomePrune] Applying Homeostatic Pruning at {target_sparsity*100:.1f}% Sparsity...")
    
    total_linear_params = 0
    total_pruned = 0
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            # Target projections in each Transformer block
            projections = {
                f"model.layers.{layer_idx}.self_attn.q_proj": layer.self_attn.q_proj,
                f"model.layers.{layer_idx}.self_attn.k_proj": layer.self_attn.k_proj,
                f"model.layers.{layer_idx}.self_attn.v_proj": layer.self_attn.v_proj,
                f"model.layers.{layer_idx}.self_attn.o_proj": layer.self_attn.o_proj,
                f"model.layers.{layer_idx}.mlp.gate_proj": layer.mlp.gate_proj,
                f"model.layers.{layer_idx}.mlp.up_proj": layer.mlp.up_proj,
                f"model.layers.{layer_idx}.mlp.down_proj": layer.mlp.down_proj,
            }
            
            for name, proj in projections.items():
                W = proj.weight # [Out_Dim, In_Dim]
                total_linear_params += W.numel()
                
                if name in activation_norms:
                    act_norm = activation_norms[name].to(W.device) # [In_Dim]
                    # Saliency score: Weight magnitude scaled by column activation norm
                    saliency = torch.abs(W) * act_norm.unsqueeze(0)
                else:
                    saliency = torch.abs(W)
                    
                # Attention Hub Protection: Attention layers pruned lightly (15%), MLPs absorb 30%
                if "self_attn" in name:
                    layer_sparsity = target_sparsity * 0.65 # Light pruning on critical routing heads
                else:
                    layer_sparsity = target_sparsity * 1.15 # Deeper pruning on redundant MLP neurons
                    
                k_val = int(W.numel() * layer_sparsity)
                if k_val > 0:
                    thresh = torch.kthvalue(saliency.flatten(), k_val).values
                    mask = (saliency > thresh).float()
                    
                    # Compute Homeostatic Scaling Factor:
                    # In biology, when synapses are pruned, surviving synapses scale up conductance
                    # to keep the layer output variance constant.
                    surviving_fraction = mask.sum().item() / W.numel()
                    homeostatic_scale = 1.0 / math.sqrt(max(0.1, surviving_fraction))
                    
                    # Apply Mask and Homeostatic Scaling
                    W.mul_(mask).mul_(homeostatic_scale)
                    total_pruned += (W.numel() - int(mask.sum().item()))
                    
    actual_sparsity = (total_pruned / total_linear_params) * 100.0
    print(f"[ConnectomePrune] Pruning complete: {actual_sparsity:.2f}% of parameters removed with homeostatic scaling.")
    return actual_sparsity

def run_calibrated_benchmark():
    print("=" * 70)
    print("HOMEORSTATIC CALIBRATED CONNECTOME-PRUNING BENCHMARK")
    print(f"Model Architecture: {MODEL_NAME} (LLaMA-3 Native Substrate)")
    print("=" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    
    test_prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "Artificial intelligence can be defined as",
        "In a biological nervous system, neurons communicate via"
    ]
    
    print("\n[Step 1] Baseline Dense Model Generations:")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(f"  -> \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")
        
    # Step 2: Biological calibration
    print("\n[Step 2] Observing Live Neural Activity across Calibration Corpus...")
    act_norms = collect_activation_norms(model, tokenizer, CALIBRATION_CORPUS)
    
    # Step 3: Apply Connectome Pruning with Homeostatic Plasticity
    print("\n[Step 3] Executing Calibrated Connectome-Pruning...")
    sparsity_achieved = apply_homeostatic_connectome_prune(model, act_norms, target_sparsity=0.25)
    
    # Step 4: Evaluate Post-Pruning Intelligence
    print("\n[Step 4] Evaluating Calibrated Pruned Model Generations:")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(f"  -> \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")
        
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    run_calibrated_benchmark()
