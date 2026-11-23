import time
import torch
from typing import List, Dict

class ModelEvaluator:
    """Evaluates latency, generation accuracy, and memory metrics of LLMs."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_responses(self, prompts: List[str], max_tokens: int = 35) -> List[Dict]:
        """Generates deterministic greedy completions and profiles latency."""
        self.model.eval()
        outputs_list = []

        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            t1 = time.perf_counter()

            gen_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
            elapsed_s = t1 - t0
            ms_per_token = (elapsed_s / max(1, new_tokens)) * 1000

            outputs_list.append({
                "prompt": p,
                "completion": gen_text,
                "new_tokens": new_tokens,
                "elapsed_ms": elapsed_s * 1000,
                "ms_per_token": ms_per_token
            })

        return outputs_list

    def compute_perplexity(self, text: str) -> float:
        """Computes cross-entropy loss and perplexity on evaluation text."""
        self.model.eval()
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        return float(torch.exp(torch.tensor(loss)).item())
