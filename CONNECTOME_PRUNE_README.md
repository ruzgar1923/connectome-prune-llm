# ConnectomePrune-LLM: Biological Connectomic Sparsification & Sharded Sculpting for Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/)

> **Author:** Rüzgar Vatansever  
> **Affiliation:** Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Turkey  
> **Contact:** `rzvatansevertr@gmail.com`  
> **Google Scholar:** [Rüzgar Vatansever Profile](https://scholar.google.com/citations?user=iqSiuPwAAAAJ)  
> **Preprint Reference:** [Zenodo DOI: 10.5281/zenodo.24375969](https://doi.org/10.5281/zenodo.24375969)

---

## 1. Overview & Motivation

Modern Large Language Models (LLMs) such as **LLaMA-3, Qwen, and Mistral** require immense computational resources, burning hundreds of gigawatts and demanding multi-GPU supercomputers for deployment. In contrast, the adult fruit fly (*Drosophila melanogaster*) executes complex navigation, odor tracking, and real-time decision-making using only **139,255 neurons and ~50 million synapses operating on mere microwatts of power** (*Nature*, 2024).

Standard neural network pruning fails catastrophically on LLMs:
- **Naive Magnitude Pruning** destroys attention routing hubs, leading to activation explosion and degenerate token repetitions (`"the the the..."`).
- **Standard Post-Training Pruning (Wanda / SparseGPT)** requires massive GPU memory to prune 70B models, locking out researchers without enterprise-grade clusters.

**ConnectomePrune-LLM** resolves both bottlenecks through two biologically grounded innovations:
1. **Biological Hub Protection & Homeostatic Synaptic Scaling:** Protects central routing hubs (analogous to the Drosophila Central Complex and Mushroom Body calyx) and automatically applies biological variance normalization ($\gamma = 1 / \sqrt{\text{surviving\_ratio}}$) to maintain downstream layer dynamics.
2. **ConnectomeSculptor (Sharded Sequential Pruning):** Inspired by classical sculpture, giant 7B/70B models are carved block-by-block (`.safetensors` shards). Each shard is loaded, pruned, scaled, saved to disk, and evicted from memory immediately. **This enables pruning 70B-parameter LLMs on ordinary consumer PCs without running out of RAM.**

---

## 2. Benchmark Results

Tested on native **LLaMA-3 architecture (`HuggingFaceTB/SmolLM-135M`)**:

| Metric | Dense Baseline | Naive Pruning (25%) | ConnectomePrune-LLM (26.07%) |
| :--- | :--- | :--- | :--- |
| **Linear Parameters** | 106,168,320 | 106,168,320 | **78,492,329** (27,675,991 pruned) |
| **Pruning Speed** | N/A | Instant | **1.91 seconds** |
| **RAM Footprint (Sculptor)** | 100% of Model | 100% of Model | **Single-Shard (~1.5 GB)** |
| **Capital of France?** | Paris | Repetition crash | **Paris** (100% Factual) |
| **Water Boiling Point?** | 100°C | Repetition crash | **100°C** (100% Factual) |
| **AI Definition?** | Coherent | Garbage tokens | **Coherent & Precise** |

---

## 3. Architecture & Key Modules

```
connectome_prune_llm/
├── __init__.py           # Package exports
├── pruner.py             # Activation-aware Saliency & Homeostatic Scaling Engine
├── sculptor.py           # Sharded Safetensors Sequential Sculptor for 7B/70B
├── evaluator.py          # Latency (ms/token) & Factual Verification Suite
└── cli.py                # Command-Line Interface
```

### The Sculptor Engine (`sculptor.py`)
```python
from connectome_prune_llm.sculptor import ConnectomeSculptor

sculptor = ConnectomeSculptor(
    target_sparsity=0.25,        # 25% synapses removed
    hub_protection_ratio=0.20,   # Attention routing hubs protected
    homeostatic_scaling=True     # Variance normalization enabled
)

# Sequentially carves any multi-gigabyte Hugging Face model directory:
sculptor.sculpt_model_directory(
    source_dir="./meta-llama-3-8b",
    target_dir="./pruned-llama-3-8b-connectome"
)
```

---

## 4. Quick Start CLI

```bash
# Clone the repository
git clone https://github.com/RuzgarVatansever/connectome-prune-llm.git
cd connectome-prune-llm

# Run pruning on any Hugging Face model
python -m connectome_prune_llm.cli --model "HuggingFaceTB/SmolLM-135M" --sparsity 0.25
```

---

## 5. Citation

If you use ConnectomePrune-LLM in your academic research, please cite:

```bibtex
@article{vatansever2026connectome,
  title={ConnectomePrune: Biological Hub-Preserving Sparsification for Billion-Scale Large Language Models},
  author={Vatansever, R{\"u}zgar},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.24375969}
}
```
