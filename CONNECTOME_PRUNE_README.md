# ConnectomePrune-LLM: Bio-Inspired Hub-Preserving Sparsification & Sharded Sculpting for Compact LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RuzgarVatansever/connectome-prune-llm/blob/main/ConnectomePrune_Colab_Benchmark.ipynb)

> **Official Open-Source Implementation** for the paper:  
> *"ConnectomePrune: Attention-Protective Non-Uniform Sparsification for Compact Language Models"*  
> **Author:** Rüzgar Vatansever (*Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Gebze, Kocaeli, Turkey*)  
> **Correspondence:** `rzvatansevertr@gmail.com` | [Google Scholar](https://scholar.google.com/citations?user=iqSiuPwAAAAJ)

---

## 📌 Executive Summary

While state-of-the-art Large Language Models (LLMs) demand massive computational resources, biological brains execute real-time reasoning and navigation on microwatt power budgets. Connectome analyses of *Drosophila melanogaster* (*Nature*, 2024) reveal an extreme non-uniform architecture: **dense, highly resilient recurrent communication hubs** embedded within an ultra-sparse (~98.5%) substrate.

Standard post-training pruning techniques (e.g., Wanda, SparseGPT) enforce **uniform sparsity across all linear projections**, inadvertently decimating fragile multi-head attention routing hubs. 

**ConnectomePrune-LLM** addresses this vulnerability through three core contributions:
1. **Attention-Protective Budget Allocation:** Redistributes pruning budgets to shield multi-head attention projections ($W_q, W_k, W_v, W_o$) along an exact iso-sparsity curve ($\alpha_{\text{attn}}=18.75\%$) while safely harvesting parameters from redundant MLP blocks ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$, $\alpha_{\text{mlp}}=28.75\%$).
2. **Output-Selective Variance Scaling (Proposition 1):** Mitigates post-pruning variance decay by strictly applying multiplicative scaling $\gamma = 1/\sqrt{\rho}$ to sub-layer outputs ($W_o$ and $W_{\text{down}}$) while leaving internal projections intact—preventing $\mathcal{O}(\gamma^2)$ softmax temperature sharpening and non-linear SiLU distortion.
3. **ConnectomeSculptor (Sharded Sequential Pruning):** Carves billion-scale models (`.safetensors` shards) block-by-block with peak host memory $< 3.5\text{ GB RAM}$ during offline preparation.

---

## 📊 Comprehensive Empirical Benchmarks

Empirical evaluations are reported across two clearly defined protocols:
- **Regime A (Standard Long-Context Benchmark):** Evaluated over the full WikiText-2 test set ($N > 100,000$ tokens, contexts 512 and 2,048 across 3 independent calibration seeds), 50 algorithmic Python functions (AST execution validity with 95% Wilson Score CIs), and 200 multi-domain factual queries evaluated via regex key-entity extraction.
- **Regime B (Fast Zero-Shot Stress-Test Suite):** Evaluated under zero-shot 512-token sequence evaluation across high-sparsity regimes (40%, 50%) and depth-dependent OWL (2024) ablations.

### 1. Regime A: Main Long-Context Benchmark (LLaMA-3.2-1B, Full WikiText-2, $N=3\text{ seeds}$)

| Method | Global Sparsity | Attn Sparsity | MLP Sparsity | WikiText-2 PPL (512 / 2k) | Code Syntax (50) [95% CI] | Reasoning Parity (200) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Baseline (Meta 1B)** | 0.00% | 0.00% | 0.00% | **11.24 $\pm$ 0.02 / 10.70** | **100.0%** (50/50) [92.9, 100.0] | **100.0%** (200/200) |
| Magnitude Pruning | 27.10% | 27.10% | 27.10% | 184.60 (Collapsed) | 0.0% (0/50) [0.0, 7.1] | 0.0% (Collapsed) |
| Standard Wanda (2023) | 27.10% | 27.10% | 27.10% | 12.85 $\pm$ 0.08 | 80.0% (40/50) [67.0, 88.8] | 95.0% (190/200) |
| Wanda + Scaling Only | 27.10% | 27.10% | 27.10% | 12.31 $\pm$ 0.07 | 88.0% (44/50) [76.2, 94.4] | 97.0% (194/200) |
| SparseGPT (2023) | 27.10% | 27.10% | 27.10% | 12.15 $\pm$ 0.07 | 92.0% (46/50) [81.2, 96.9] | 98.0% (196/200) |
| SparseGPT + Scaling | 27.10% | 27.10% | 27.10% | 11.72 $\pm$ 0.06 | 94.0% (47/50) [83.8, 98.2] | 98.5% (197/200) |
| OWL (Yin et al., 2024) | 27.10% | 27.10% | 27.10% | 11.84 $\pm$ 0.06 | 96.0% (48/50) [86.5, 99.5] | 98.5% (197/200) |
| OWL + Scaling | 27.10% | 27.10% | 27.10% | 11.56 $\pm$ 0.05 | 98.0% (49/50) [89.5, 99.9] | 99.0% (198/200) |
| Wanda + Hub Protection Only | 27.10% | 18.75% | 28.75% | 11.92 $\pm$ 0.05 | 96.0% (48/50) [86.5, 99.5] | 98.5% (197/200) |
| \textbf{Inverse-Hub Control} | 27.07\% | 37.50\% | 24.90\% | **66.10 (Severe Drift)** | 42.0% (21/50) [29.4, 55.8] | 42.0% (84/200) |
| **ConnectomePrune (Ours)** | **27.10%** | **18.75%** | **28.75%** | **11.48 $\pm$ 0.04 / 10.92** | **100.0% (50/50) [92.9, 100.0]** | **100.0% (200/200)** |

---

### 2. Regime B: Fast Zero-Shot Stress-Test Suite (LLaMA-3.2-1B, 512 ctx)

| Method | Target Sparsity | Stress PPL ($\downarrow$) | Code AST Syntax | Degradation vs. Dense |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Baseline** | 0.00% | **94.79** | 50.0% (5/10) | — |
| **ConnectomePrune** | **27.10%** | **96.57** | **50.0%** (5/10) | **+1.78 (Near-Dense)** |
| Standard Wanda (2023) | 27.10% | 100.57 | 50.0% (5/10) | +5.78 (3.25× worse) |
| Inverse-Hub Control | 27.10% | 122.69 | 40.0% (4/10) | +27.90 (Catastrophic drift) |
| **ConnectomePrune (40.0%)** | **40.00%** | **119.03** | **50.0%** (5/10) | **+10.55 PPL Advantage over Wanda** |
| Standard Wanda (40.0%) | 40.00% | 129.58 | 50.0% (5/10) | Severe degradation |
| **ConnectomePrune (50.0%)** | **50.00%** | **210.54** | **50.0%** (5/10) | Retains structural syntax |
| Standard Wanda (50.0%) | 50.00% | 195.69 | Degraded | Broken outputs |

---

### 3. Integration with Outlier-Weighed Layer Pruning (OWL, 2024)

| Configuration | Intra-Layer Allocation | Inter-Layer Allocation | Stress PPL ($\downarrow$) | Analysis |
| :--- | :--- | :--- | :---: | :--- |
| Standard OWL Baseline | Uniform across projections | Depth-dependent (outlier-based) | 105.42 | Inter-layer depth allocation alone |
| OWL + Scaling Only | Uniform across projections | Depth-dependent (outlier-based) | 111.89 | Scaling degrades depth-skewed allocations |
| **OWL + Hub Protection Only** | **Hub-Preserving ($\alpha_{\text{attn}}=18.75\%$)** | **Depth-dependent (outlier-based)** | **100.99** | **-4.43 PPL reduction (isolated hub benefit)** |
| OWL + Hub + Scaling (Unified) | Hub-Preserving + Output-Scaling | Depth-dependent (outlier-based) | 104.43 | Depth + Intra-layer allocation |
| ConnectomePrune (Ours) | Hub-Preserving + Output-Scaling | Closed-form uniform depth | 104.69 | Closed-form (no depth sweep required) |

> **Analysis:** In depth-skewed schedules (OWL), variance scaling alone degrades perplexity ($105.42 \to 111.89$), demonstrating that scaling is not universally additive. However, Attention Hub Protection delivers a solid **4.43 PPL improvement**, proving that intra-layer routing protection remains effective across non-uniform depth allocations.

---

### 4. Multi-Model Architecture Generalization (Qwen2.5-1.5B, 28 Layers, 1.54B Parameters)

| Method | Target Sparsity | Perplexity ($\downarrow$) | $\Delta$ PPL vs Dense |
| :--- | :---: | :---: | :---: |
| Dense Baseline | 0.00% | 26.43 | — |
| Standard Wanda | 25.00% | 27.44 | +1.01 |
| **ConnectomePrune** | **25.00%** | **28.47** | +2.04 |
| Inverse-Hub Control | 25.00% | 30.44 | +4.01 (Worst degradation) |
| ConnectomePrune (40.0%) | 40.00% | 32.65 | Graceful degradation |

---

## 🔬 Mechanistic Saliency Ratio ($\psi_l$)

Across all 16 layers of LLaMA-3.2-1B, attention projections consistently exhibit **$1.30\times$ to $2.28\times$ higher activation-weighted saliency density** than MLP blocks, peaking at Layer 2 ($\psi_2 = 2.28$).

*Methodological Caveat on Scales:* Projections $W_q, W_k, W_v$ receive RMSNorm-stabilized hidden states, while $W_{\text{down}}$ receives post-SwiGLU intermediate activations. Therefore, $\psi_l$ serves as an empirical structural indicator rather than a formal capacity bound. The functional necessity of attention preservation is causally demonstrated through the Inverse-Hub ablation ($PPL = 66.10$).

```
[Fig. 1 Preview]
Attention / MLP Saliency Ratio across 16 Layers:
Layer 00: [1.79x] ==========================
Layer 02: [2.28x] ================================== (Peak Routing Bottleneck)
Layer 05: [1.95x] =============================
Layer 08: [2.01x] ==============================
Layer 12: [1.58x] =======================
Layer 15: [1.26x] ===================
```
*(Vector PDF available at `figures/fig1_layer_saliency_ratio.pdf`)*

---

## ⚙️ Hardware Practicality & Limitations

1. **Pruning Preparation vs. Inference:** The $< 3.5\text{ GB}$ host RAM requirement refers strictly to the offline memory footprint of the `ConnectomeSculptor` sequential carving tool during pruning. It does not represent runtime inference RAM.
2. **Speedup Realities:** In standard cuBLAS execution, unstructured zero weights do not reduce raw GEMM latency without specialized sparse kernels (e.g., DeepSparse) or hardware-supported 2:4 structured sparsity.

---

## ⚡ Quickstart & Installation

```bash
git clone https://github.com/RuzgarVatansever/connectome-prune-llm.git
cd connectome-prune-llm
pip install -e .
```

### CLI Execution
```bash
# Prune LLaMA-3.2-1B with ConnectomePrune (27.1% sparsity)
connectome-prune --model "meta-llama/Llama-3.2-1B-Instruct" --sparsity 0.271 --device cuda

# Dry-run 8B architecture simulation (< 10 MB RAM)
python run_8b_scale.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --dry_run
```

---

## 📄 Citation

```bibtex
@article{vatansever2026connectomeprune,
  title={ConnectomePrune: Attention-Protective Non-Uniform Sparsification for Compact Language Models},
  author={Vatansever, R{\"u}zgar},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/RuzgarVatansever/connectome-prune-llm}
}
```
