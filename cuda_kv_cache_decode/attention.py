import torch
import math
import custom_flash_attention
import custom_flash_attention_decode
import torch.nn.functional as F

class CustomFlashAttention(torch.nn.Module):
    '''
    Custom FlashAttention implementation.

    Args:
        w_q: Query weight matrix of shape (hidden_dim, hidden_dim)
        w_k: Key weight matrix of shape (hidden_dim, hidden_dim)
        w_v: Value weight matrix of shape (hidden_dim, hidden_dim)
        w_o: Output weight matrix of shape (hidden_dim, hidden_dim)
        num_heads: Number of attention heads
        b_q, b_k, b_v, b_o: Optional bias vectors of shape (hidden_dim,)
    '''
    def __init__(self, w_q, w_k, w_v, w_o, hidden_dim, num_heads, 
                 b_q=None, b_k=None, b_v=None, b_o=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o
        self.b_q = b_q
        self.b_k = b_k
        self.b_v = b_v
        self.b_o = b_o

    def forward(self, x, kv_cache, decode=False, current_pos=0, causal=False):
        '''
        Forward pass for the self-attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            decode: Whether to use decode mode
            kv_cache: Tuple of (k_cache, v_cache) of shape (batch_size, max_seq_len, hidden_dim)
            current_pos: Current position in the cache (for decode mode)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        '''
    
        batch_size, seq_len, _ = x.shape
        k_cache, v_cache = kv_cache

        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T
        
        # Add biases if present
        if self.b_q is not None:
            q = q + self.b_q
        if self.b_k is not None:
            k = k + self.b_k
        if self.b_v is not None:
            v = v + self.b_v

        if decode:
            assert x.shape[1] == 1, "x must have shape (batch_size, 1, hidden_dim) in decode mode"

            # Update the kv_cache at current_pos
            custom_flash_attention_decode.update_kv_cache(k_cache, v_cache, k, v, current_pos)
            
            # Use the updated cache for attention (up to current_pos + 1)
            # Extract the relevant portion of the cache and make it contiguous
            k_full = k_cache[:, :current_pos + 1, :].contiguous()
            v_full = v_cache[:, :current_pos + 1, :].contiguous()
            
            # Perform attention with full context
            out = custom_flash_attention_decode.custom_flash_attention_decode(q, k_full, v_full, self.num_heads, causal)

        else:
            # Store the kv_cache
            k_cache[:, current_pos:current_pos + seq_len, :] = k
            v_cache[:, current_pos:current_pos + seq_len, :] = v

            out = custom_flash_attention.custom_flash_attention(q, k, v, self.num_heads, causal)

        # Apply output projection
        out = out @ self.w_o.T
        if self.b_o is not None:
            out = out + self.b_o
        return out