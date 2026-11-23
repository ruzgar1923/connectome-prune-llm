"""
Multi-Domain Stress Test & Validation Suite for Pruned Qwen-1.5B
==============================================================
Tests 10 diverse tasks:
1. Mathematics: Arithmetic
2. Coding: Python syntax
3. Logic: Deductive reasoning
4. Geography: World capitals
5. Biology: Cellular structure
6. Physics: Classical mechanics
7. History: Major events
8. Language: Grammar completion
9. Chemistry: Periodic table
10. Computer Science: Data structures
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_CASES = [
    {"domain": "Mathematics", "prompt": "Solve: 15 + 27 ="},
    {"domain": "Coding", "prompt": "In Python, the keyword to define a function is"},
    {"domain": "Logic", "prompt": "All humans are mortal. Socrates is a human. Therefore, Socrates is"},
    {"domain": "Geography", "prompt": "The Great Wall is located in the country of"},
    {"domain": "Biology", "prompt": "The organelle known as the powerhouse of the cell is the"},
    {"domain": "Physics", "prompt": "Isaac Newton is famously associated with the discovery of the law of universal"},
    {"domain": "History", "prompt": "The first man to walk on the moon was Neil"},
    {"domain": "Language", "prompt": "The opposite of hot is"},
    {"domain": "Chemistry", "prompt": "The chemical symbol for gold is"},
    {"domain": "Computer Science", "prompt": "A Last-In, First-Out (LIFO) data structure is called a"}
]

def run_stress_test(model_path):
    print("=" * 70)
    print("MULTI-DOMAIN STRESS TEST ON PRUNED MODEL")
    print(f"Path: {model_path}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()

    results = []

    for idx, test in enumerate(TEST_CASES, 1):
        prompt = test["prompt"]
        domain = test["domain"]
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        completion = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract only the continuation
        continuation = completion[len(prompt):].strip()
        print(f"[{domain}] Prompt: '{prompt}'")
        print(f"  -> Model Continuation: '{continuation}'\n")

        results.append({
            "domain": domain,
            "prompt": prompt,
            "full_text": completion,
            "continuation": continuation
        })

    return results

if __name__ == "__main__":
    path = "C:/Users/hp/.gemini/antigravity/scratch/drosophila-vs-ann-benchmark/sculpted_qwen_1.5b"
    run_stress_test(path)
