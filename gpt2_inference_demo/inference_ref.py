import os
import torch

os.makedirs("/tmp/huggingface", exist_ok=True)
os.environ["HF_HOME"] = "/tmp/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to("cuda").eval()

prompts = [
    "What are the advantages of using flashattention kernels compared to the naive pytorch implementation? Think about the shared memory usage, the computation complexity, and the performance."
]
    
for prompt in prompts:
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.95,
            top_k=10,
            do_sample=True
        )
    
    # Decode and print
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("="*60)
    print(f"\nGenerated text:\n{generated_text}\n")
    print("="*60)