# FlashAttention LLM Inference with CUDA

End-to-end LLM inference engine with custom CUDA FlashAttention kernels, tiled GEMM operations, online softmax, KV caching, and optimized prefill/decode pipelines.

## Introduction

Large Language Models (LLMs) rely on the Transformer architecture, which uses self-attention as its fundamental operation. In this project, I implemented highly optimized CUDA kernels for attention computation, specifically targeting the FlashAttention algorithm, along with a complete end-to-end LLM inference pipeline.

## Background

### Self-Attention

Attention computes the relationship between Query (Q), Key (K), and Value (V) matrices:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

where $d$ is the head dimension. The scaling factor prevents dot products from growing too large and saturating the softmax.

In **self-attention**, Q, K, and V are all derived from the same input sequence $X$ through learned linear projections:
- $Q = X \cdot W_Q$
- $K = X \cdot W_K$  
- $V = X \cdot W_V$

### Multi-Head Attention

Instead of a single attention operation, **Multi-Head Attention (MHA)** splits Q, K, V into $h$ heads, computes attention independently for each head, then concatenates results:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O
$$

The input tensor is reshaped from `[batch, seq_len, hidden_dim]` to `[batch, num_heads, seq_len, head_dim]` where `head_dim = hidden_dim / num_heads`.

### Causal Masking

For autoregressive generation, tokens can only attend to previous positions. This is achieved by masking the upper triangle of the attention scores with $-\infty$ before softmax.

## Implementations

### 1. PyTorch Multi-Head Attention (`pytorch_multihead_attention/`)

A baseline implementation of multi-head self-attention in pure PyTorch. The implementation reshapes Q, K, V tensors to separate heads, computes scaled dot-product attention with optional causal masking, applies softmax, and projects the output.

```bash
python -m pytorch_multihead_attention.test
```
**Output:** Compares against `torch.nn.MultiheadAttention` and reports whether outputs match within tolerance.

---

### 2. CUDA Multi-Head Attention (`cuda_multihead_attention/`)

Custom CUDA kernels implementing the attention operation from scratch:

**Softmax Kernel:** Numerically stable softmax using the max-subtraction trick. Uses shared memory for parallel reduction to find row maximum, then computes `exp(x - max)` and normalizes. Each CUDA block handles one row vector.

**Tiled GEMM Kernels:** Two separate kernels optimized for different transpose configurations:
- `GEMM_NT`: Computes $A \times B^T$ for the $QK^T$ operation
- `GEMM_NN`: Computes $A \times B$ for the attention-weighted value computation

Both use shared memory tiling to maximize memory bandwidth utilization.

**Attention Pipeline:** Chains the kernels together: GEMM_NT → scale & causal mask → softmax → GEMM_NN.

```bash
python -m cuda_multihead_attention.compile 
python -m cuda_multihead_attention.test --softmax  # Tests softmax kernel correctness
python -m cuda_multihead_attention.test --gemm     # Tests GEMM kernels correctness
python -m cuda_multihead_attention.test --attention # Tests full attention pipeline
```
**Output:** Reports pass/fail for each kernel and compares against PyTorch reference.

---

### 3. PyTorch FlashAttention (`pytorch_flash_attention/`)

Implementation of the **FlashAttention-2** algorithm following [Algorithm 1 from Tri Dao's paper](https://arxiv.org/abs/2307.08691). 

The key innovation is **tiled computation with online softmax**: instead of materializing the full $N \times N$ attention matrix, the algorithm processes Q and KV in blocks while maintaining running statistics (max and sum) to compute correct softmax values incrementally.

The algorithm maintains per-row accumulators:
- $m_i$: running maximum for numerical stability
- $l_i$: running sum of exponentials (softmax denominator)
- $O_i$: running weighted output

For each new KV block, it updates: $m_{new} = \max(m_{old}, \max(S_{block}))$, rescales previous accumulations by $e^{m_{old} - m_{new}}$, and adds the new block's contribution.

```bash
python -m pytorch_flash_attention.test
```
**Output:** Validates correctness against naive attention and reports timing comparison.

---

### 4. CUDA FlashAttention (`cuda_flash_attention/`)

The FlashAttention algorithm translated into a **fused CUDA kernel**. This performs the entire attention computation (both GEMMs, masking, softmax, and output projection) in a single kernel launch, keeping intermediate results in shared memory.

**Kernel Design:**
- Grid: `(batch_size × num_heads, num_query_blocks)` — each block handles one batch-head and one query tile
- Block: `B_r` threads — each thread handles one row in the query tile
- Shared memory: Tiles for Q, K, V, output, plus running statistics (m, l) and score buffers

**Memory Optimization:** By processing in tiles of size `B_r × B_c`, the kernel avoids materializing the $O(N^2)$ attention matrix in global memory. Tile sizes are tuned to fit within the 48KB shared memory limit per block.

```bash
python -m cuda_flash_attention.compile
python -m cuda_flash_attention.test
```
**Output:** Reports speedup vs PyTorch for both causal and non-causal attention at different batch/sequence sizes. Achieves up to **7.78x speedup** on large sequences.

---

### 5. CUDA KV Cache Decode (`cuda_kv_cache_decode/`)

Optimized kernels for the **decode phase** of LLM inference using KV caching.

**The Problem:** During autoregressive generation, each new token requires attending to all previous tokens. Naively, this means recomputing K and V for the entire sequence at each step.

**KV Cache Solution:** Pre-allocate memory for K and V up to `max_seq_len`. During prefill, store all K/V values. During decode, only compute K/V for the single new token and append to the cache.

**Implemented Kernels:**

1. **`update_cache_kernel`:** Copies new K and V vectors (single token) into the appropriate cache position. Simple but critical for avoiding memory allocation overhead.

2. **`flash_attention_decode_kernel`:** Specialized FlashAttention for `seq_len=1` queries. Since Q is always a single row, this eliminates the outer loop over Q blocks. Each thread block handles one batch-head, iterating only over KV blocks.

```bash
python -m cuda_kv_cache_decode.compile
python -m cuda_kv_cache_decode.test
```
**Output:** Runs prefill on 1024 tokens, then generates 100 tokens using decode kernel. Reports TTFT (Time To First Token) and TBT (Time Between Tokens) speedups. Achieves **3.6x TTFT** and **3.1x TBT** improvement.

---

### 6. LLM Inference Pipeline (`llm_inference/`)

Complete **end-to-end LLM inference** implementation combining all components into a working transformer decoder.

**Architecture:**
- RoPE Embedding layer
- N decoder layers, each containing:
  - Multi-head self-attention with RoPE (Rotary Position Embedding)
  - SwiGLU MLP (gate, up, down projections with SiLU activation)
  - Layer normalization (pre-norm style)
- Output projection to vocabulary

**Inference Modes:**
- **Prefill:** Full FlashAttention kernel processes the input prompt, stores K/V in cache
- **Decode:** Specialized decode kernel generates tokens one at a time using cached K/V

The attention module automatically detects the mode based on input sequence length and manages cache updates.

```bash
python -m llm_inference.test
```
**Output:** Simulates full generation loop (1024 token prefill + 100 token decode). Reports prefill time (TTFT) and average decode time (TBT) in milliseconds.

---

### 7. GPT-2 Inference Demo (`gpt2_inference_demo/`)

Demonstrates the custom kernels integrated with a **real Hugging Face GPT-2 model**. The custom attention kernels replace the standard attention computation while using GPT-2's pretrained weights.

```bash
python -m gpt2_inference_demo.inference      # Uses custom CUDA kernels
python -m gpt2_inference_demo.inference_ref  # Uses standard HuggingFace attention
```
**Output:** Generates text from a prompt, allowing comparison between custom and reference implementations to verify correctness.

---

## Performance Results

See `Results.md` for  benchmarks.
---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Improved parallelism and work partitioning
- [KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)

---

### Acknowledgments

Based on CS 8803 GPU course project (Georgia Tech)
