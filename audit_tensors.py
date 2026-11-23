"""
Comprehensive Tensor-Level Numerical Audit
==========================================
Checks all tensors in the pruned model for:
1. NaN or Inf values
2. Exact zero counts (true sparsity)
3. Layer-by-layer variance and mean drift
4. Weight norm stability
"""

import os
import torch
from safetensors.torch import load_file

def audit_weights(original_path, pruned_path):
    print("=" * 70)
    print("STEP 1: TENSOR-LEVEL NUMERICAL INTEGRITY AUDIT")
    print(f"Original: {original_path}")
    print(f"Pruned:   {pruned_path}")
    print("=" * 70)

    orig_tensors = load_file(original_path)
    pruned_tensors = load_file(pruned_path)

    total_tensors = len(pruned_tensors)
    nan_count = 0
    inf_count = 0
    mismatched_shapes = 0
    total_elements = 0
    total_zeros = 0

    print(f"Total Tensors in Model: {total_tensors}")

    layer_stats = []

    for name, p_tensor in pruned_tensors.items():
        if name not in orig_tensors:
            print(f"[ERROR] Missing tensor in original: {name}")
            continue

        o_tensor = orig_tensors[name]

        # 1. Shape check
        if p_tensor.shape != o_tensor.shape:
            print(f"[CRITICAL] Shape mismatch in {name}: {p_tensor.shape} vs {o_tensor.shape}")
            mismatched_shapes += 1

        # 2. NaN / Inf check
        has_nan = torch.isnan(p_tensor).any().item()
        has_inf = torch.isinf(p_tensor).any().item()
        if has_nan:
            nan_count += 1
            print(f"[CRITICAL] NaN detected in {name}!")
        if has_inf:
            inf_count += 1
            print(f"[CRITICAL] Inf detected in {name}!")

        # 3. Sparsity check
        numel = p_tensor.numel()
        total_elements += numel
        zeros = (p_tensor == 0).sum().item()
        total_zeros += zeros

        # If it's a linear projection layer, track statistics
        if "weight" in name and p_tensor.dim() == 2:
            sparsity = (zeros / numel) * 100.0
            o_std = torch.std(o_tensor.float()).item()
            p_std = torch.std(p_tensor.float()).item()
            o_mean = torch.mean(o_tensor.float()).item()
            p_mean = torch.mean(p_tensor.float()).item()
            
            layer_stats.append({
                "name": name,
                "shape": list(p_tensor.shape),
                "sparsity": sparsity,
                "orig_std": o_std,
                "pruned_std": p_std,
                "std_ratio": p_std / (o_std + 1e-8),
                "orig_mean": o_mean,
                "pruned_mean": p_mean
            })

    print("\n--- Numerical Audit Summary ---")
    print(f"Total Tensors Verified: {total_tensors}")
    print(f"NaN Values Detected:    {nan_count} (Pass if 0)")
    print(f"Inf Values Detected:    {inf_count} (Pass if 0)")
    print(f"Shape Mismatches:       {mismatched_shapes} (Pass if 0)")
    print(f"Total Model Parameters: {total_elements:,}")
    print(f"Exact Zeros (Pruned):   {total_zeros:,}")
    print(f"True Net Sparsity:      {(total_zeros / total_elements) * 100:.2f}%")

    print("\n--- Sample Layer Variance Audit (First 5 Linear Layers) ---")
    for s in layer_stats[:5]:
        print(f"Layer: {s['name']}")
        print(f"  Shape: {s['shape']} | Sparsity: {s['sparsity']:.2f}%")
        print(f"  Original Std: {s['orig_std']:.6f} | Pruned Std: {s['pruned_std']:.6f} (Ratio: {s['std_ratio']:.3f})")
        print(f"  Original Mean: {s['orig_mean']:.6f} | Pruned Mean: {s['pruned_mean']:.6f}")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mismatches": mismatched_shapes,
        "total_params": total_elements,
        "total_zeros": total_zeros,
        "true_sparsity": (total_zeros / total_elements) * 100.0
    }

if __name__ == "__main__":
    orig_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/model.safetensors")
    pruned_path = "C:/Users/hp/.gemini/antigravity/scratch/drosophila-vs-ann-benchmark/sculpted_qwen_1.5b/model.safetensors"
    audit_weights(orig_path, pruned_path)
