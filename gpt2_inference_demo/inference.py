"""
Simple inference script using custom flash attention kernels with HuggingFace GPT-2 models.
"""
import torch
import sys
import os

os.makedirs("/tmp/huggingface", exist_ok=True)
os.environ["HF_HOME"] = "/tmp/huggingface"

from transformers import AutoTokenizer, AutoModelForCausalLM
from task5.attention import CustomFlashAttention

class ModelWithCustomAttention:
    """Wrapper to replace GPT-2 model's attention with custom kernels"""
    
    def __init__(self, model_name="gpt2", max_seq_len=2048):
        print(f"Loading GPT-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model dimensions
        hidden_dim = self.model.config.n_embd
        num_heads = self.model.config.n_head
        
        print(f"Model loaded: {hidden_dim} hidden_dim, {num_heads} heads")
        
        # Replace attention modules with custom implementation
        self._replace_attention_modules()
    
    def _replace_attention_modules(self):
        """Replace standard attention with custom flash attention"""
        hidden_dim = self.model.config.n_embd
        num_heads = self.model.config.n_head
        layers = self.model.transformer.h
        
        print(f"\nReplacing {len(layers)} attention layers with custom kernels...")
        
        for layer_idx, block in enumerate(layers):
            # GPT-2 uses combined c_attn for Q, K, V
            attn = block.attn
            qkv_weight = attn.c_attn.weight.data
            qkv_bias = attn.c_attn.bias.data if attn.c_attn.bias is not None else None
            
            w_q = qkv_weight[:, :hidden_dim].T.contiguous()
            w_k = qkv_weight[:, hidden_dim:2*hidden_dim].T.contiguous()
            w_v = qkv_weight[:, 2*hidden_dim:].T.contiguous()
            w_o = attn.c_proj.weight.data.T.contiguous()
            
            if qkv_bias is not None:
                b_q = qkv_bias[:hidden_dim]
                b_k = qkv_bias[hidden_dim:2*hidden_dim]
                b_v = qkv_bias[2*hidden_dim:]
            else:
                b_q = b_k = b_v = None
            
            b_o = attn.c_proj.bias.data if attn.c_proj.bias is not None else None
            
            # Create custom attention module
            custom_attn = CustomFlashAttention(
                w_q, w_k, w_v, w_o, 
                hidden_dim, num_heads,
                b_q, b_k, b_v, b_o
            ).to("cuda").eval()
            
            # Store reference to custom attention
            block.custom_attn = custom_attn
        
        print("✓ All attention layers replaced with custom kernels")
    
    def _forward_with_custom_attention(self, input_ids, use_cache=False, past_key_values=None, current_pos=0):
        """Forward pass using custom attention"""
        batch_size, seq_len = input_ids.shape
        
        hidden_dim = self.model.config.n_embd
        layers = self.model.transformer.h
        
        # GPT-2 embeddings
        hidden_states = self.model.transformer.wte(input_ids)
        if seq_len == 1:
            position_ids = torch.tensor([[current_pos]], device="cuda")
        else:
            position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        hidden_states = hidden_states + self.model.transformer.wpe(position_ids)
        
        # Initialize caches if needed (prefill)
        if use_cache and past_key_values is None:
            past_key_values = []
            for _ in range(len(layers)):
                k_cache = torch.zeros(batch_size, self.max_seq_len, hidden_dim, 
                                     dtype=torch.float32, device="cuda")
                v_cache = torch.zeros(batch_size, self.max_seq_len, hidden_dim,
                                     dtype=torch.float32, device="cuda")
                past_key_values.append((k_cache, v_cache))
        
        # Determine mode
        decode_mode = use_cache and seq_len == 1
        
        # Process through transformer blocks
        for layer_idx, block in enumerate(layers):
            # GPT-2: layer norm before attention
            hidden_states_norm = block.ln_1(hidden_states)
            attn_output = block.custom_attn(
                hidden_states_norm,
                kv_cache=past_key_values[layer_idx],
                decode=decode_mode,
                current_pos=current_pos,
                causal=True
            )
            hidden_states = hidden_states + attn_output
            
            # GPT-2: MLP
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
        
        # Final layer norm and LM head
        hidden_states = self.model.transformer.ln_f(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        return logits, past_key_values
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
        """Generate text using custom attention kernels"""
        print(f"\nPrompt: {prompt}")
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        print(f"Prompt tokens: {input_ids.shape[1]}")
        
        if input_ids.shape[1] == 0:
            print("ERROR: Empty input_ids! Prompt may be too long or tokenization failed.")
            return ""
        
        # Prefill phase
        print("\n[Prefill] Processing prompt...")
        logits, past_key_values = self._forward_with_custom_attention(
            input_ids, use_cache=True, past_key_values=None, current_pos=0
        )
        
        # Get first generated token
        next_token_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        current_pos = input_ids.shape[1]  # Position for next token
        
        # Decode phase
        print(f"[Decode] Generating {max_new_tokens - 1} tokens...")
        max_context = self.max_seq_len
        
        for step in range(max_new_tokens - 1):
            # Check if we've reached max context length
            if current_pos >= max_context:
                print(f"\nReached maximum context length ({max_context}), stopping generation.")
                break
            
            # Generate next token
            logits, past_key_values = self._forward_with_custom_attention(
                next_token, use_cache=True, past_key_values=past_key_values, current_pos=current_pos
            )
            
            current_pos += 1
            
            next_token_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print("="*60)
        print(f"\nGenerated text:\n{generated_text}")
        
        return generated_text

def main():
    """Main inference function
    
    Available GPT-2 models:
    - gpt2 (124M parameters)
    - gpt2-medium (355M parameters)
    - gpt2-large (774M parameters)
    - gpt2-xl (1.5B parameters)
    - distilgpt2 (smaller, faster)
    """
    model_name = "gpt2"  # GPT-2 base model
    max_seq_len = 2048

    # Create model with custom attention
    model = ModelWithCustomAttention(model_name, max_seq_len)
    
    # Example prompts
    prompts = [
        "What are the advantages of using flashattention kernels compared to the naive pytorch implementation? Think about the shared memory usage, the computation complexity, and the performance."
    ]
    
    # Generate for each prompt
    for prompt in prompts:
        model.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.95,
            top_k=10
        )
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

