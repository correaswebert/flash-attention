import torch

class RoPE(torch.nn.Module):
    def __init__(self, head_dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for max_seq_len
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len):
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    
    # Reshape cos and sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    # Split x into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # Apply rotation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_out = x * cos + rotated * sin
    
    return x_out