import math
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ConnectomePruner:
    """
    Bio-inspired Non-Uniform LLM Pruning Engine:
    - Identifies Attention Routing Hubs (Central Complex principle) and protects them.
    - Allocates heavy sparsity to redundant Feed-Forward projections (Kenyon Cell principle).
    - Applies Homeostatic Synaptic Scaling to maintain layer-wise variance.
    """
    def __init__(
        self,
        model,
        tokenizer,
        target_sparsity: float = 0.25,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_sparsity = target_sparsity
        self.device = device
        self.activation_norms: Dict[str, torch.Tensor] = {}

    def collect_activations(self, calibration_corpus: List[str]) -> None:
        """Collects empirical input activation norms ||X_j||_2 across calibration sequences."""
        self.activation_norms.clear()
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                x = inp[0].detach()
                x_flat = x.reshape(-1, x.shape[-1])
                norm = torch.norm(x_flat, p=2, dim=0)
                if name in self.activation_norms:
                    self.activation_norms[name] += norm.cpu()
                else:
                    self.activation_norms[name] = norm.cpu().clone()
            return hook_fn

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            for text in calibration_corpus:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.model(**inputs)

        for h in hooks:
            h.remove()

    def prune_layer_linear(
        self,
        linear_module: nn.Linear,
        layer_name: str,
        sparsity_ratio: float,
        is_attention: bool = False,
        n_heads: int = 1
    ) -> int:
        """
        Prunes a single linear projection matrix using saliency = |W| * ||X||
        and applies biological homeostatic scaling.
        """
        W = linear_module.weight.data
        out_dim, in_dim = W.shape
        numel = W.numel()

        # Retrieve empirical activation norm
        if layer_name in self.activation_norms:
            act_norm = self.activation_norms[layer_name].to(W.device)
            if act_norm.shape[0] == in_dim:
                saliency = torch.abs(W) * act_norm.unsqueeze(0)
            else:
                saliency = torch.abs(W)
        else:
            saliency = torch.abs(W)

        # Non-uniform attention head hub preservation
        if is_attention and n_heads > 1:
            head_dim = out_dim // n_heads
            total_pruned_in_module = 0

            # Measure per-head Frobenius norm centrality
            head_centralities = []
            for h in range(n_heads):
                h_slice = W[h * head_dim : (h + 1) * head_dim, :]
                head_centralities.append(torch.norm(h_slice, p='fro').item())

            c_tensor = torch.tensor(head_centralities)
            norm_c = (c_tensor - c_tensor.min()) / (c_tensor.max() - c_tensor.min() + 1e-6)

            # High centrality heads get lower sparsity budget
            head_budgets = sparsity_ratio * (0.7 + 0.6 * (1.0 - norm_c))

            for h in range(n_heads):
                budget = head_budgets[h].item()
                h_w = W[h * head_dim : (h + 1) * head_dim, :]
                h_s = saliency[h * head_dim : (h + 1) * head_dim, :]
                k_val = int(h_w.numel() * budget)

                if k_val > 0:
                    thresh = torch.kthvalue(h_s.flatten(), k_val).values
                    mask = (h_s > thresh).float()
                    surviving_ratio = mask.sum().item() / h_w.numel()
                    h_scale = 1.0 / math.sqrt(max(0.1, surviving_ratio))
                    h_w.mul_(mask).mul_(h_scale)
                    total_pruned_in_module += (h_w.numel() - int(mask.sum().item()))

            return total_pruned_in_module
        else:
            # Standard FFN channel pruning
            k_val = int(numel * sparsity_ratio)
            if k_val <= 0:
                return 0

            thresh = torch.kthvalue(saliency.flatten(), k_val).values
            mask = (saliency > thresh).float()
            surviving_ratio = mask.sum().item() / numel
            h_scale = 1.0 / math.sqrt(max(0.1, surviving_ratio))
            W.mul_(mask).mul_(h_scale)
            return numel - int(mask.sum().item())

    def execute_pruning(self) -> Dict[str, float]:
        """
        Executes layer-by-layer Connectome-Pruning across the entire model.
        Supports LLaMA, Mistral, Qwen, SmolLM, and Gemma architectures.
        """
        start_time = time.time()
        total_linear_params = 0
        total_pruned_params = 0

        # Detect architecture layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError("Unsupported Transformer architecture. Ensure model has .layers attribute.")

        n_layers = len(layers)
        n_heads = getattr(self.model.config, "num_attention_heads", 8)

        with torch.no_grad():
            for layer_idx, layer in enumerate(layers):
                # Attention Projections: Protected Hub regime
                attn_sparsity = self.target_sparsity * 0.70
                mlp_sparsity = self.target_sparsity * 1.15

                # Discover linear submodules in current layer
                for mod_name, mod in layer.named_modules():
                    if isinstance(mod, nn.Linear):
                        full_name = f"model.layers.{layer_idx}.{mod_name}" if "model.layers" not in mod_name else mod_name
                        total_linear_params += mod.weight.numel()

                        is_attn = any(k in mod_name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"])
                        spec_sparsity = attn_sparsity if is_attn else mlp_sparsity

                        n_pruned = self.prune_layer_linear(
                            mod,
                            layer_name=full_name,
                            sparsity_ratio=spec_sparsity,
                            is_attention=is_attn,
                            n_heads=n_heads
                        )
                        total_pruned_params += n_pruned

        elapsed = time.time() - start_time
        overall_sparsity = (total_pruned_params / total_linear_params) * 100.0

        return {
            "total_linear_params": total_linear_params,
            "pruned_params": total_pruned_params,
            "sparsity_pct": overall_sparsity,
            "elapsed_seconds": elapsed,
        }

    def save_pruned_model(self, output_dir: str):
        """Saves pruned model and tokenizer to disk."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[ConnectomePrune] Successfully saved model to {output_dir}")
