import torch

from .attention import CustomFlashAttention

def main(causal=False):
    hidden_dim = 128
    num_heads = 8

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

        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        custom_out = custom_self_attention(x, causal=causal)

        assert torch.allclose(torch_out, custom_out, atol=1e-6)

    batch_size = 16
    seq_len = 4096

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        if causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        else:
            attn_mask = None

        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        custom_out = custom_self_attention(x, causal=causal)

        assert torch.allclose(torch_out, custom_out, atol=1e-6)

def test():
    main()
    main(causal=True)

    print("All tests passed!")

if __name__ == "__main__":
    test()