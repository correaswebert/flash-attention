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

        scale = 1.0 / math.sqrt(self.head_dim)

        B_r = (seq_len + T_r - 1) // T_r
        B_c = (seq_len + T_c - 1) // T_c

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2.
        o = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim,
                        device="cuda", dtype=torch.float)

        # 3.
        for i in range(T_r):
            # 4.
            q_start = i * B_r
            q_end = min(q_start + B_r, seq_len)
            Q_i = q[:, :, q_start:q_end, :]

            # 5.
            O_i = torch.zeros(batch_size, self.num_heads, q_end - q_start, self.head_dim,
                              device="cuda", dtype=torch.float)
            l_i = torch.zeros(batch_size, self.num_heads, q_end - q_start,
                              device="cuda", dtype=torch.float)
            m_i = torch.full((batch_size, self.num_heads, q_end - q_start),
                             -1e9,
                             device="cuda", dtype=torch.float)

            # 6.
            for j in range(T_c):
                # 7.
                kv_start = j * B_c
                kv_end = min(kv_start + B_c, seq_len)
                K_j = k[:, :, kv_start:kv_end, :]
                V_j = v[:, :, kv_start:kv_end, :]

                # 8.
                S_i = Q_i @ K_j.mT * scale

                if causal:
                    query_positions = torch.arange(q_start, q_end, device="cuda").unsqueeze(1)
                    key_positions = torch.arange(kv_start, kv_end, device="cuda").unsqueeze(0)

                    causal_mask = (query_positions >= key_positions).unsqueeze(0)

                    S_i = S_i.masked_fill(~causal_mask, -1e9)

                m_i_prev = m_i
                l_i_prev = l_i

                # 9.
                m_i = torch.maximum(m_i_prev, S_i.max(dim=-1).values)
                correction_factor = torch.exp(m_i_prev - m_i)
                P_i = torch.exp(S_i - m_i.unsqueeze(-1))
                l_i = correction_factor * l_i_prev + P_i.sum(dim=-1)

                # 10.
                O_i = correction_factor.unsqueeze(-1) * O_i + P_i @ V_j

            # 12.
            O_i = O_i / l_i.unsqueeze(-1)

            # 14.
            o[:, :, q_start:q_end, :] = O_i

        # Reshape back (batch_size, seq_len, hidden_dim)
        o = o.transpose(1, 2)
        o = o.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 17.
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