"""
Connectome Sculptor: Sharded / Chunked LLM Pruning Engine
=========================================================
Implements Rüzgar Vatansever's 'Sculptor' principle:
Pruning colossal LLMs (7B, 13B, 70B) by carving individual safetensor shards
sequentially without ever exhausting system RAM.

Author: Rüzgar Vatansever & Antigravity
"""

import os
import gc
import json
import math
import shutil
import time
import torch
from typing import Dict, List, Optional
from safetensors.torch import load_file, save_file

class ConnectomeSculptor:
    """
    Sequentially carves and prunes large LLM weight shards (7B, 70B)
    using biological connectome principles and homeostatic variance scaling.
    """
    def __init__(
        self,
        target_sparsity: float = 0.25,
        hub_protection_ratio: float = 0.20,
        homeostatic_scaling: bool = True
    ):
        self.target_sparsity = target_sparsity
        self.hub_protection_ratio = hub_protection_ratio
        self.homeostatic_scaling = homeostatic_scaling

    def sculpt_tensor(self, tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, int, int]:
        """
        Sculpts a single 2D weight matrix:
        - Excludes 1D biases and layernorm scales.
        - Identifies high-centrality synaptic hubs and protects them.
        - Prunes low-saliency connections.
        - Restores homeostatic signal variance.
        """
        if tensor.dim() != 2:
            return tensor, 0, tensor.numel()

        # Crucial LLM Invariant: Never prune token embeddings, lm_head, or norms
        # Vocabulary vectors and output logits must remain 100% intact to prevent token distortion.
        if any(k in name.lower() for k in ["embed", "lm_head", "wte", "wpe", "norm"]):
            return tensor, 0, tensor.numel()

        numel = tensor.numel()
        out_features, in_features = tensor.shape

        # Identify if this is a routing hub (e.g. self-attention Q, K, V, O)
        is_attention_hub = any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"])
        
        # Effective sparsity budget (Feed-Forward layers pruned more, Attention hubs protected)
        if is_attention_hub:
            eff_sparsity = self.target_sparsity * (1.0 - self.hub_protection_ratio)
        else:
            eff_sparsity = min(0.40, self.target_sparsity * 1.15)

        k_prune = int(numel * eff_sparsity)
        if k_prune <= 0:
            return tensor, 0, numel

        # Synaptic Saliency magnitude
        abs_w = torch.abs(tensor)
        kth_val = torch.kthvalue(abs_w.view(-1), k_prune).values.item()

        # Binary synaptic mask
        mask = (abs_w > kth_val).to(tensor.dtype)
        pruned_count = (mask == 0).sum().item()

        # Homeostatic Synaptic Scaling (Biological compensation)
        if self.homeostatic_scaling:
            surviving_ratio = max(0.1, (numel - pruned_count) / numel)
            gamma = 1.0 / math.sqrt(surviving_ratio)
        else:
            gamma = 1.0

        sculpted_tensor = tensor * mask * gamma
        return sculpted_tensor, pruned_count, numel

    def sculpt_shard(self, shard_path: str, output_path: str) -> Dict[str, float]:
        """Loads a single safetensors shard, carves it, and saves it immediately."""
        t0 = time.perf_counter()
        tensors = load_file(shard_path)
        
        sculpted_dict = {}
        total_params = 0
        total_pruned = 0

        for name, tensor in tensors.items():
            sculpted_w, pruned_cnt, orig_cnt = self.sculpt_tensor(tensor, name)
            sculpted_dict[name] = sculpted_w
            total_pruned += pruned_cnt
            total_params += orig_cnt

        save_file(sculpted_dict, output_path)
        
        # Free memory immediately
        del tensors
        del sculpted_dict
        gc.collect()

        elapsed = time.perf_counter() - t0
        sparsity_pct = (total_pruned / max(1, total_params)) * 100.0

        return {
            "shard": os.path.basename(shard_path),
            "total_params": total_params,
            "pruned_params": total_pruned,
            "sparsity_pct": sparsity_pct,
            "elapsed_seconds": elapsed
        }

    def sculpt_model_directory(self, source_dir: str, target_dir: str) -> Dict:
        """
        Sculpts an entire multi-shard model directory sequentially.
        """
        os.makedirs(target_dir, exist_ok=True)
        shard_files = [f for f in os.listdir(source_dir) if f.endswith(".safetensors")]
        
        print("=" * 70)
        print("CONNECTOME SCULPTOR: SHARDED SEQUENTIAL PRUNING")
        print(f"Source Directory: {source_dir}")
        print(f"Target Directory: {target_dir}")
        print(f"Total Shards to Sculpt: {len(shard_files)}")
        print("=" * 70)

        shard_reports = []
        cum_total = 0
        cum_pruned = 0

        for i, shard_name in enumerate(shard_files, 1):
            src_shard = os.path.join(source_dir, shard_name)
            dst_shard = os.path.join(target_dir, shard_name)
            print(f"\n[Sculpting Shard {i}/{len(shard_files)}]: {shard_name} ...")
            
            report = self.sculpt_shard(src_shard, dst_shard)
            cum_total += report["total_params"]
            cum_pruned += report["pruned_params"]
            shard_reports.append(report)
            
            print(f"  -> Done in {report['elapsed_seconds']:.2f}s | "
                  f"Pruned: {report['pruned_params']:,} / {report['total_params']:,} "
                  f"({report['sparsity_pct']:.2f}% sparse)")

        # Copy metadata files (config.json, tokenizer.json, index files)
        for fname in os.listdir(source_dir):
            if not fname.endswith(".safetensors"):
                src_f = os.path.join(source_dir, fname)
                dst_f = os.path.join(target_dir, fname)
                if os.path.isfile(src_f):
                    shutil.copy2(src_f, dst_f)

        net_sparsity = (cum_pruned / max(1, cum_total)) * 100.0
        print("\n" + "=" * 70)
        print("SCULPTING COMPLETE!")
        print(f"Total Model Parameters: {cum_total:,}")
        print(f"Total Synapses Removed: {cum_pruned:,}")
        print(f"Net Connectome Sparsity: {net_sparsity:.2f}%")
        print("=" * 70)

        return {
            "total_params": cum_total,
            "pruned_params": cum_pruned,
            "net_sparsity": net_sparsity,
            "shards": shard_reports
        }
