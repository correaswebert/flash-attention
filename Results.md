### Task 4

Non-causal self-attention speedups:
- batch_size = 4, seq_len = 128: 2.59
- batch_size = 64, seq_len = 4096: 7.78

Causal self-attention speedups:
- batch_size = 4, seq_len = 128: 1.75
- batch_size = 64, seq_len = 4096: 2.15

### Task 5

Prefill & Decode speedups:
- Time to first token (TTFT): 3.60
- Time between tokens (TBT): 3.12

### Task 6

Prefill & Decode times (ms):
- Prefill time (TTFT): 2851.68 ms
- Average decode time (TBT): 51.88 ms

### Generated Output

Generated output:

What are the advantages of using flashattention kernels compared to the naive pytorch implementation? Think about the shared memory usage, the computation complexity, and the performance. The naive pytorch implementation will consume a lot more memory, and is much more memory-intensive. The naive pytorch implementation is only about 2% of the original size, which is a good result, but not much of a big deal


