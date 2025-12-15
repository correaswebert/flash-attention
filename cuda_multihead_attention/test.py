import torch
import torch.nn.functional as F
import argparse

from .attention import CustomSelfAttention

def test_softmax():
    import custom_self_attention

    batch_size = 4

    with torch.no_grad():
        x = torch.randn(batch_size, 10, 10).to(torch.float).to("cuda")
        torch_out = F.softmax(x, dim=-1)
        custom_out = custom_self_attention.test_batched_softmax(x, -1)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        x = torch.randn(batch_size, 5, 5).to(torch.float).to("cuda")
        torch_out = F.softmax(x, dim=-2)
        custom_out = custom_self_attention.test_batched_softmax(x, -2)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        x = torch.randn(batch_size, 1000, 1000).to(torch.float).to("cuda")
        torch_out = F.softmax(x, dim=-1)
        custom_out = custom_self_attention.test_batched_softmax(x, -1)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        print("Softmax test passed!")

def test_gemm():
    import custom_self_attention

    batch_size = 4

    with torch.no_grad():
        a = torch.randn(batch_size, 10, 10).to(torch.float).to("cuda")
        b = torch.randn(batch_size, 10, 10).to(torch.float).to("cuda")
        torch_out = a @ b
        custom_out = custom_self_attention.test_batched_GEMM_NN(a, b)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)
        torch_out = a @ b.transpose(-2, -1)
        custom_out = custom_self_attention.test_batched_GEMM_NT(a, b)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        a = torch.randn(batch_size, 1000, 1000).to(torch.float).to("cuda")
        b = torch.randn(batch_size, 1000, 1000).to(torch.float).to("cuda")
        torch_out = a @ b
        custom_out = custom_self_attention.test_batched_GEMM_NN(a, b)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)
        torch_out = a @ b.transpose(-2, -1)
        custom_out = custom_self_attention.test_batched_GEMM_NT(a, b)
        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        print("GEMM test passed!")

def test_attention(causal=False):
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
        CustomSelfAttention(w_q, w_k, w_v, w_o, hidden_dim, num_heads)
        .to(torch.float)
        .to("cuda")
        .eval()
    )

    batch_size = 4
    seq_len = 10

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
    seq_len = 100
    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim).to(torch.float).to("cuda")

        if causal:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to("cuda")
        else:
            attn_mask = None

        torch_out, _ = torch_self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        custom_out = custom_self_attention(x, causal=causal)

        assert torch.allclose(torch_out, custom_out, atol=1e-6)

        if causal:
            print("Causal Self-Attention test passed!")
        else:
            print("Non-causal Self-Attention test passed!")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--softmax", action="store_true", default=False, help="Test the softmax function")
    args.add_argument("--gemm", action="store_true", default=False, help="Test the GEMM function")
    args.add_argument("--attention", action="store_true", default=False, help="Test the attention function")
    args = args.parse_args()

    if args.softmax:
        test_softmax()

    elif args.gemm:
        test_gemm()

    elif args.attention:
        test_attention()
        test_attention(causal=True)

    else:
        test_softmax()
        test_gemm()
        test_attention()
        test_attention(causal=True)

        print("All tests passed!")