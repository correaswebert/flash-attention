import torch
import math
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
    '''
    def __init__(self, w_q, w_k, w_v, w_o, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o

    def _flash_attention(self, q, k, v, T_r=8, T_c=8, causal=False):
        '''
        FlashAttention implementation.

        Args:
            q: Query tensor of shape (batch_size, seq_len, hidden_dim)
            k: Key tensor of shape (batch_size, seq_len, hidden_dim)
            v: Value tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        '''
    
        batch_size, seq_len, _ = q.shape

        # Reshape and flatten heads into shape (batch_size * num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim)

        # Output and streaming softmax accumulators
        o = torch.zeros(batch_size * self.num_heads, seq_len, self.head_dim, device="cuda", dtype=torch.float)

        scale = 1.0 / math.sqrt(self.head_dim)
    
        # Number of blocks
        num_q_blocks = (seq_len + T_r - 1) // T_r
        num_kv_blocks = (seq_len + T_c - 1) // T_c

        for i in range(num_q_blocks):
            q_start = i * T_r
            q_end=min(q_start + T_r, seq_len)
            Q_i=q[:, q_start:q_end, :]
            m_i=torch.full((batch_size * self.num_heads, q_end - q_start), 
                         -torch.inf, device="cuda", dtype=torch.float)
            l_i=torch.zeros((batch_size * self.num_heads, q_end - q_start), device="cuda", dtype=torch.float)
            O_i=torch.zeros((batch_size * self.num_heads, q_end - q_start, self.head_dim), device="cuda", dtype=torch.float)
            
            for j in range(num_kv_blocks):
                kv_start = j * T_c
                kv_end = min(kv_start + T_c, seq_len)
                K_j = k[:, kv_start:kv_end, :] # shape: (batch*heads, block_size, head_dim)
                V_j = v[:, kv_start:kv_end, :]
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # (batch*heads, q_block, kv_block)

                if causal:
                    query_positions = torch.arange(q_start, q_end, device="cuda")
                    key_positions = torch.arange(kv_start, kv_end, device="cuda")
                    query_positions_2d = query_positions.unsqueeze(1)
                    key_positions_2d = key_positions.unsqueeze(0)
                    causal_mask = query_positions_2d >= key_positions_2d 
                    causal_mask = causal_mask.unsqueeze(0)
                    S_ij = S_ij.masked_fill(~causal_mask, -torch.inf)
                    
                m_i_new=torch.maximum(m_i, S_ij.max(dim=-1).values)# (BH, Tr)
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))# (BH, Tr, Tc) broadcast
                l_i_new = torch.exp(m_i - m_i_new) * l_i + P_ij.sum(dim=-1)
                correction_factor = torch.exp(m_i - m_i_new)
                correction_factor_expanded = correction_factor.unsqueeze(-1)  # (BH, Tr) to (BH, Tr, 1)
                O_i_rescaled = O_i * correction_factor_expanded         # (BH, Tr, d) * (BH, Tr, 1) to (BH, Tr, d)  [broadcast]
                P_times_V = torch.matmul(P_ij, V_j)                     # (BH, Tr, Tc) @ (BH, Tc, d) to (BH, Tr, d)
                O_i = O_i_rescaled + P_times_V                          # (BH, Tr, d) + (BH, Tr, d) to(BH, Tr, d)
                m_i = m_i_new
                l_i = l_i_new

            O_i = O_i / l_i.unsqueeze(-1)
            o[:, q_start:q_end, :] = O_i




       
        # Reshape back (batch_size, seq_len, hidden_dim)
        o = o.view(batch_size, self.num_heads, seq_len, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)
        
        return o

    def forward(self, x, causal=False):
        '''
        Forward pass for the self-attention layer using FlashAttention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        '''
    
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        o = self._flash_attention(q, k, v, causal=causal)

        o = o @ self.w_o.T
        return o