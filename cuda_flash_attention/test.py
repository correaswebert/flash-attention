import torch
import torch.nn.functional as F
import time
from .attention import CustomFlashAttention

def test_attention(causal=False):
    hidden_dim = 4096
    num_heads = 32

    torch_self_attention = (
        torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, bias=False, batch_first=True)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    all_param = dict(torch_self_attention.named_parameters())
    w_q, w_k, w_v = all_param["in_proj_weight"].chunk(3)
    w_o = all_param["out_proj.weight"]

    custom_self_attention = (
        CustomFlashAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    batch_size = 4
    seq_len = 128

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        if causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        else:
            attn_mask = None

        # warm up
        _, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        _ = custom_self_attention(x, causal=causal)

        start_time = time.time()
        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        torch_time = time.time() - start_time
        start_time = time.time()
        custom_out = custom_self_attention(x, causal=causal)
        custom_time = time.time() - start_time
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        print(f"{'' if causal else 'Non-'}Causal Self-Attention: batch_size = {batch_size}, seq_len = {seq_len}")
        print(f"Torch time (ms): {torch_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup: {torch_time / custom_time :.2f}")

    batch_size = 64
    seq_len = 4096

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        if causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        else:
            attn_mask = None

        # warm up
        _, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        _ = custom_self_attention(x, causal=causal)
        
        start_time = time.time()
        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        torch_time = time.time() - start_time
        start_time = time.time()
        custom_out = custom_self_attention(x, causal=causal)
        custom_time = time.time() - start_time

        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        print(f"{'' if causal else 'Non-'}Causal Self-Attention: batch_size = {batch_size}, seq_len = {seq_len}")
        print(f"Torch time (ms): {torch_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup: {torch_time / custom_time :.2f}")

    if causal:
        print("Causal Self-Attention test passed!")
    else:
        print("Non-causal Self-Attention test passed!")

if __name__ == "__main__":
    torch.manual_seed(41)
    test_attention()
    test_attention(causal=True)

    print("All tests passed!")