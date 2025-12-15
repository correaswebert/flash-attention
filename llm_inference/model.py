import custom_flash_attention
import custom_flash_attention_decode
import torch
import math

from .rope import RoPE, apply_rotary_emb

class customModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, vocab_size, max_seq_len=4096):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        assert hidden_dim // num_heads <= 128, f"head_dim ({hidden_dim / num_heads}) must be less than or equal to 128"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList([customDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.ln = torch.nn.LayerNorm(hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, kv_cache=None, current_pos=0):
        """
        Args:
            x: Input tokens of shape (batch_size, seq_len)
            kv_cache: Optional list of (k_cache, v_cache) tuples for each layer
            current_pos: Current position for decode mode
        """
        batch_size, seq_len = x.shape
        decode = seq_len == 1 and kv_cache is not None
        
        # Initialize KV cache if not provided
        if kv_cache is None:
            kv_cache = []
            for _ in range(self.num_layers):
                k_cache = torch.zeros(batch_size, self.max_seq_len, self.hidden_dim, 
                                     dtype=torch.float32, device=x.device)
                v_cache = torch.zeros(batch_size, self.max_seq_len, self.hidden_dim,
                                     dtype=torch.float32, device=x.device)
                kv_cache.append((k_cache, v_cache))
        
        x = self.embedding(x)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, kv_cache=kv_cache[layer_idx], decode=decode, current_pos=current_pos)
        x = self.ln(x)
        x = self.lm_head(x)
        return x, kv_cache

class customDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = customAttention(hidden_dim, num_heads)
        self.mlp = customMLP(hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, kv_cache=None, decode=False, current_pos=0):
        x = x + self.attention(self.norm1(x), kv_cache=kv_cache, decode=decode, current_pos=current_pos)
        x = x + self.mlp(self.norm2(x))
        return x

class customAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.w_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.w_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.w_o = torch.nn.Linear(hidden_dim, hidden_dim)
        self.rope = RoPE(self.head_dim)

    def forward(self, x, kv_cache=None, decode=False, current_pos=0):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape for multi-head attention: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        if decode:
            # In decode mode, use current_pos for RoPE
            cos, sin = self.rope(q, current_pos + 1)
            cos = cos[current_pos:current_pos+1, :]  # (1, head_dim)
            sin = sin[current_pos:current_pos+1, :]  # (1, head_dim)
        else:
            cos, sin = self.rope(q, current_pos + seq_len)
            cos = cos[current_pos:current_pos+seq_len, :]  # (seq_len, head_dim)
            sin = sin[current_pos:current_pos+seq_len, :]  # (seq_len, head_dim)
        
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Reshape back to (batch, seq, hidden) for flash attention
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        k = k.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        v = v.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Apply flash attention with KV caching
        if decode and kv_cache is not None:
            ##############################################################

            k_cache, v_cache = kv_cache
            custom_flash_attention_decode.update_kv_cache(k_cache, v_cache, k, v, current_pos)
            k_full = k_cache[:, :current_pos + 1, :].contiguous()
            v_full = v_cache[:, :current_pos + 1, :].contiguous()
            o = custom_flash_attention_decode.custom_flash_attention_decode(
                q, k_full, v_full, self.num_heads, True  
            )

            ##############################################################
        else:
            ##############################################################

            k_cache, v_cache = kv_cache
            
            k_cache[:, current_pos:current_pos + seq_len, :] = k
            v_cache[:, current_pos:current_pos + seq_len, :] = v
            
            o = custom_flash_attention.custom_flash_attention(
                q, k, v, self.num_heads, True  # causal=True for decoder
            )

            ##############################################################
        
        o = self.w_o(o)
        return o

class customMLP(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_gate = torch.nn.Linear(hidden_dim, hidden_dim * 4)
        self.w_up = torch.nn.Linear(hidden_dim, hidden_dim * 4)
        self.w_out = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        self.swish = torch.nn.SiLU()
    def forward(self, x):
        gate = self.swish(self.w_gate(x))
        up = self.w_up(x)
        return self.w_out(gate * up)