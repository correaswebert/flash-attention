#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace torch::indexing;

// FlashAttention kernel
template <int B_r, int B_c>
__global__ void flashattention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles
    // Max head_dim = 128
    // +8 for bank conflict resolution (even for vectorized loads like float4)
    __shared__ float q_tile[B_r][128 + 8];
    __shared__ float k_tile[B_c][128 + 8];
    __shared__ float v_tile[B_c][128 + 8];
    __shared__ float o_tile[B_r][128 + 8];

    // for online softmax
    __shared__ float m_tile[B_r]; // running max of the S row
    __shared__ float l_tile[B_r]; // running sum of the S row

    int T_c = (seq_len + B_c - 1) / B_c;

    // 3.
    int bh_idx = blockIdx.x; // batch_size * num_heads
    int i = blockIdx.y; // i in 0..T_r

    int q_tile_start = i * B_r;

    // if (q_tile_start >= seq_len) return;

    // each thread handles one row in the q block
    int q_tile_row_idx = threadIdx.x; // 0..B_r
    int q_row_idx = q_tile_start + q_tile_row_idx;

    // get one head from multi-heads
    const float *q_bh = q + (bh_idx * seq_len) * head_dim;
    const float *k_bh = k + (bh_idx * seq_len) * head_dim;
    const float *v_bh = v + (bh_idx * seq_len) * head_dim;
    float *o_bh = out + (bh_idx * seq_len) * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    // 4.
    int q_tile_size = B_r * head_dim;

    for (int idx = threadIdx.x; idx < q_tile_size; idx += blockDim.x) {
        int local_row_idx = idx / head_dim;
        int d_idx = idx % head_dim;

        int global_row_idx = q_tile_start + local_row_idx;
    
        q_tile[local_row_idx][d_idx] = (global_row_idx < seq_len)
                                        ? q_bh[global_row_idx * head_dim + d_idx]
                                        : 0.0f;

        // 5.
        o_tile[local_row_idx][d_idx] = 0.0f;
    }

    // 5.
    for (int row = threadIdx.x; row < B_r; row += blockDim.x) {
        l_tile[row] = 0.0f;
        m_tile[row] = -FLT_MAX;
    }

    __syncthreads();

    for (int j = 0; j < T_c; j++) {
        int kv_tile_start = j * B_c;
        int kv_tile_size = B_c * head_dim;

        // B_r - 1 buffer for the tiles on the diagonal
        if (causal && kv_tile_start > q_tile_start + (B_r - 1)){
            break;
        }

        // 7.
        for (int idx = threadIdx.x; idx < kv_tile_size; idx += blockDim.x) {
            int local_row_idx = idx / head_dim;
            int d_idx = idx % head_dim;
            
            int global_row_idx = kv_tile_start + local_row_idx;

            k_tile[local_row_idx][d_idx] = (global_row_idx < seq_len)
                                            ? k_bh[global_row_idx * head_dim + d_idx]
                                            : 0.0f;
        }

        for (int idx = threadIdx.x; idx < kv_tile_size; idx += blockDim.x) {
            int local_row_idx = idx / head_dim;
            int d_idx = idx % head_dim;
            
            int global_row_idx = kv_tile_start + local_row_idx;

            v_tile[local_row_idx][d_idx] = (global_row_idx < seq_len)
                                            ? v_bh[global_row_idx * head_dim + d_idx]
                                            : 0.0f;
        }

        __syncthreads();
        
        // Each thread works on one row of the q_tile
        for (int row = threadIdx.x; row < B_r; row += blockDim.x) {
            float m_i_prev = m_tile[row];
            float l_i_prev = l_tile[row];
        
            for (int col = 0; col < B_c; col++) {
                float score = 0.0f;

                if (causal && ((q_tile_start + row) < (kv_tile_start + col))) {
                    score = -FLT_MAX;
                } else {
                    // 8. Q(i) @ K(j).mT
                    for (int d = 0; d < head_dim; d++) {
                        score += q_tile[row][d] * k_tile[col][d];
                    }

                    score *= scale;
                }
        
                // 9. Online Softmax Update
                float m_i = fmaxf(m_i_prev, score);
                float correction_factor = expf(m_i_prev - m_i);
                float p_i = expf(score - m_i);
                float l_i = (correction_factor * l_i_prev) + p_i;
        
                // 10. O(i) = correction * O(i-1) + P_i @ V_j
                for (int d = 0; d < head_dim; d++) {
                    // Rescale the old partial sum
                    float o_val = o_tile[row][d] * correction_factor;

                    // Add the new V contribution
                    o_tile[row][d] = o_val + (p_i * v_tile[col][d]);
                }
        
                m_i_prev = m_i;
                l_i_prev = l_i;
            }
            
            // Write stats back to shared memory for the next j-block
            m_tile[row] = m_i_prev;
            l_tile[row] = l_i_prev;
        }
    } // 11. end j-for

    // 12. 14.
    if (q_row_idx < seq_len){
        float l_inv = 1.0f / l_tile[q_tile_row_idx];

        for (int d = 0; d < head_dim; d++) {
            o_bh[(q_row_idx * head_dim) + d] = l_inv * o_tile[q_tile_row_idx][d];
        }
    }
}

torch::Tensor custom_flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, seq_len, head_dim)
    auto q_bh = q.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});

    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});

    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});

    auto out_bh = torch::zeros_like(q_bh, options);

    const int B_r = 16;  // Query tile size (rows)
    const int B_c = 16;  // KV tile size (columns)
    int T_r = (seq_len + B_r - 1) / B_r;
    
    // Block: B_r threads (one thread per query row in the tile)
    dim3 grid_size(batch_size * num_heads, T_r);
    dim3 block_size(B_r);
    
    flashattention_kernel<B_r, B_c><<<grid_size, block_size>>>(
        q_bh.const_data_ptr<float>(),
        k_bh.const_data_ptr<float>(),
        v_bh.const_data_ptr<float>(),
        out_bh.mutable_data_ptr<float>(),
        seq_len,
        head_dim,
        causal
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, seq_len, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, seq_len, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, seq_len, hidden_dim});
    
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention", &custom_flash_attention, "Custom FlashAttention in CUDA");
}
