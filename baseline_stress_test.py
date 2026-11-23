import os
import glob
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

pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*")
source_dir = glob.glob(pattern)[0]

print("Running baseline tests on UNPRUNED ORIGINAL MODEL...")
tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(source_dir, torch_dtype=torch.bfloat16, local_files_only=True)
model.eval()

for test in TEST_CASES:
    domain = test["domain"]
    prompt = test["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    completion = tokenizer.decode(out[0], skip_special_tokens=True)
    continuation = completion[len(prompt):].strip()
    print(f"[{domain}] Prompt: '{prompt}'")
    print(f"  -> ORIGINAL Baseline Continuation: '{continuation}'\n")
