import torch
import torch.nn.functional as F
import time 

from .attention import CustomFlashAttention

def test_attention(causal=False):
    hidden_dim = 4096
    num_heads = 32
    max_seq_len = 2048

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

    batch_size = 16
    seq_len = 1024

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")
        k_cache = torch.zeros(batch_size, max_seq_len, hidden_dim).to(torch.float).to("cuda")
        v_cache = torch.zeros(batch_size, max_seq_len, hidden_dim).to(torch.float).to("cuda")

        if causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        else:
            attn_mask = None

        # warm up
        _, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        _ = custom_self_attention(x, kv_cache=(k_cache, v_cache), current_pos=0, causal=causal)

        start_time = time.time()
        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        torch_time = time.time() - start_time
        start_time = time.time()
        custom_out = custom_self_attention(x, kv_cache=(k_cache, v_cache), current_pos=0, causal=causal)
        custom_time = time.time() - start_time

        assert torch.allclose(torch_out, custom_out, atol=1e-6), "Torch and custom prefill outputs do not match"

        print(f"{'' if causal else 'Non-'}Causal Self-Attention Prefill")
        print(f"Torch time (ms): {torch_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup on time to first token (TTFT): {torch_time / custom_time :.2f}")

        all_tokens = x

        torch_times = []
        custom_times = []

        for i in range(101):
            new_token = torch_out[:, -1:, :]
            all_tokens = torch.cat([all_tokens, new_token], dim=1)

            if causal:
                attn_mask = torch.triu(torch.ones(seq_len + i + 1, seq_len + i + 1), diagonal=1).bool().to("cuda")

            start_time = time.time()
            torch_out, _ = torch_self_attention(all_tokens, all_tokens, all_tokens, need_weights=False, attn_mask=attn_mask)
            torch_time = time.time() - start_time
            start_time = time.time()
            custom_out = custom_self_attention(new_token, kv_cache=(k_cache, v_cache), decode=True, current_pos=seq_len + i, causal=causal)
            custom_time = time.time() - start_time

            assert torch.allclose(torch_out[:, -1:, :], custom_out[:, -1:, :], atol=1e-6), "Torch and custom decode outputs do not match at step %d" % (i + 1)

            torch_times.append(torch_time)
            custom_times.append(custom_time)

        # remove the first token to exclude warmup effects
        torch_time = sum(torch_times[1:]) / len(torch_times[1:])
        custom_time = sum(custom_times[1:]) / len(custom_times[1:])

        print(f"{'' if causal else 'Non-'}Causal Self-Attention Decode ({len(torch_times) - 1} tokens)")
        print(f"Torch time (ms): {torch_time * 1000 :.2f}, Custom time (ms): {custom_time * 1000 :.2f}")
        print(f"Speedup on time between tokens (TBT): {torch_time / custom_time :.2f}")

    if causal:
        print("Causal Self-Attention test passed!")
    else:
        print("Non-causal Self-Attention test passed!")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_attention(causal=True)

    print("All tests passed!")