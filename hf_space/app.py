import os
import time
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "RuzgarVatansever/Llama-3.2-1B-ConnectomePruned"

print("Loading Connectome-Pruned LLaMA-3.2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None
).to(device)
model.eval()

def generate_chat(message, history, max_tokens, temperature):
    messages = [{"role": "system", "content": "You are a helpful, precise, and highly intelligent AI assistant powered by biological connectome-pruned LLaMA-3.2."}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=temperature > 0.05,
            temperature=max(0.01, temperature) if temperature > 0.05 else None,
            pad_token_id=tokenizer.eos_token_id
        )
    t1 = time.perf_counter()

    new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
    elapsed = t1 - t0
    tok_per_sec = new_tokens / max(1e-4, elapsed)
    ms_per_tok = (elapsed / max(1, new_tokens)) * 1000

    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    status_info = f"⚡ Generated {new_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s | {ms_per_tok:.1f} ms/token) | 263.6M Synapses Pruned (27.1% Sparsity)"
    
    return reply, status_info

custom_css = """
.gradio-container { font-family: 'Segoe UI', system-ui, sans-serif; }
#status-box { background: #111827; color: #10B981; border: 1px solid #374151; font-weight: bold; border-radius: 8px; padding: 10px; }
"""

with gr.Blocks(css=custom_css, title="ConnectomePrune-LLM Live Playground") as demo:
    gr.Markdown("""
    # 🧠 ConnectomePrune: Meta LLaMA-3.2-1B-Instruct
    ### Biological Hub-Preserving Sparsification (263 Million Weights Pruned with Zero Reasoning Loss)
    **Principal Investigator:** Rüzgar Vatansever (17-year-old student, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Turkey)  
    [GitHub Repository](https://github.com/ruzgar1923/connectome-prune-llm) | [Hugging Face Model](https://huggingface.co/RuzgarVatansever/Llama-3.2-1B-ConnectomePruned) | [Zenodo Preprint DOI: 10.5281/zenodo.24375969](https://doi.org/10.5281/zenodo.24375969)
    """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=480, label="Connectome-Pruned LLaMA-3.2 Chat")
            msg_input = gr.Textbox(placeholder="Ask anything in English or Turkish (coding, physics, logic, math)...", label="Your Message")
            with gr.Row():
                send_btn = gr.Button("Send Message", variant="primary")
                clear_btn = gr.Button("Clear Chat")
            status_display = gr.Markdown("⚡ System Ready: 263,672,581 weights pruned. Homeostatic variance preserved.", elem_id="status-box")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Telemetry & Parameters")
            max_tokens = gr.Slider(32, 256, value=100, step=8, label="Max New Tokens")
            temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature (0 = Deterministic)")
            
            gr.Markdown("""
            ### 📊 Model Architecture
            - **Base Architecture:** Meta LLaMA-3.2-1B
            - **Active Linear Weights:** 709,405,947
            - **Weights Carved:** 263,672,581
            - **Net Linear Sparsity:** 27.10%
            - **Hub Protection:** Drosophila Central Complex
            - **Variance Normalization:** $\\gamma = 1/\\sqrt{0.729} \\approx 1.17$
            """)

            gr.Examples(
                examples=[
                    "Write a clean Python function to check if a number is prime.",
                    "Explain why the sky looks blue in simple physical terms.",
                    "Calculate 15 multiplied by 14.",
                    "Merhaba! Kendini kısaca Türkçe olarak tanıtır mısın?"
                ],
                inputs=msg_input
            )

    def user_submit(user_msg, history):
        return "", history + [[user_msg, None]]

    def bot_reply(history, max_tok, temp):
        user_msg = history[-1][0]
        past_hist = history[:-1]
        bot_res, status = generate_chat(user_msg, past_hist, max_tok, temp)
        history[-1][1] = bot_res
        return history, status

    send_action = msg_input.submit(user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
        bot_reply, [chatbot, max_tokens, temperature], [chatbot, status_display]
    )
    send_btn.click(user_submit, [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
        bot_reply, [chatbot, max_tokens, temperature], [chatbot, status_display]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch()
