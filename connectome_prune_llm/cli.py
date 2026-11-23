import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .pruner import ConnectomePruner
from .evaluator import ModelEvaluator

DEFAULT_CALIBRATION = [
    "The solar system consists of the Sun and celestial bodies gravitationally bound to it.",
    "Albert Einstein formulated the theory of general relativity, fundamentally altering physics.",
    "Photosynthesis enables green plants to synthesize chemical energy using sunlight.",
    "In neuroscience, action potentials travel along axons to release synaptic neurotransmitters.",
    "Computer architecture defines the conceptual structure and functional operation of hardware systems.",
    "Natural selection acts on phenotypic variations, shaping evolutionary adaptation across generations.",
    "Thermodynamics dictates the physical limits of energy conversion and informational entropy."
]

TEST_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "Artificial intelligence can be defined as",
    "In a biological nervous system, neurons communicate via"
]

def main():
    parser = argparse.ArgumentParser(description="ConnectomePrune-LLM: Bio-Inspired Model Sparsification")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Hugging Face model ID or path")
    parser.add_argument("--sparsity", type=float, default=0.25, help="Target model sparsity ratio (e.g. 0.25 for 25%)")
    parser.add_argument("--save", type=str, default="", help="Path to save the pruned model (optional)")
    args = parser.parse_args()

    print("=" * 70)
    print("CONNECTOME-PRUNE LLM: BIO-INSPIRED SPARSITY COMPRESSION")
    print(f"Target Model: {args.model}")
    print(f"Target Sparsity: {args.sparsity * 100:.1f}%")
    print("Author: Rüzgar Vatansever | Bio-inspired AI Initiative")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Environment] Compute Device: {device.upper()}")

    print("\n[Step 1/4] Loading Pretrained Model & Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)

    evaluator = ModelEvaluator(model, tokenizer, device=device)

    print("\n[Step 2/4] Baseline Generation Quality (Unpruned Dense):")
    base_res = evaluator.generate_responses(TEST_PROMPTS)
    for r in base_res:
        print(f"  * \"{r['completion']}\" ({r['ms_per_token']:.1f} ms/token)")

    print("\n[Step 3/4] Activating Connectome Calibration & Hub Protection...")
    pruner = ConnectomePruner(model, tokenizer, target_sparsity=args.sparsity, device=device)
    pruner.collect_activations(DEFAULT_CALIBRATION)

    stats = pruner.execute_pruning()
    print(f"\n[Pruning Summary]:")
    print(f"  * Total Linear Parameters: {stats['total_linear_params']:,}")
    print(f"  * Synapses Pruned:         {stats['pruned_params']:,}")
    print(f"  * Net Sparsity Achieved:   {stats['sparsity_pct']:.2f}% REMOVED")
    print(f"  * Execution Time:          {stats['elapsed_seconds']:.2f} seconds")

    print("\n[Step 4/4] Connectome-Pruned Generation Quality:")
    pruned_res = evaluator.generate_responses(TEST_PROMPTS)
    for r in pruned_res:
        print(f"  * \"{r['completion']}\" ({r['ms_per_token']:.1f} ms/token)")

    if args.save:
        pruner.save_pruned_model(args.save)

    print("\n" + "=" * 70)
    print("CONNECTOME-PRUNE PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
