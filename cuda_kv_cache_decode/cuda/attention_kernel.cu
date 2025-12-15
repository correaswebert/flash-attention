#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

using namespace torch::indexing;

// FlashAttention kernel
template <int B>
__global__ void flash_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int seq_len,
    int head_dim,
    bool causal)
{
    // Shared memory for tiles
    __shared__ float q_tile[128];  // Max head_dim = 128
    __shared__ float k_tile[B][128];
    __shared__ float v_tile[B][128];
    __shared__ float o_tile[128];
    __shared__ float m_tile;
    __shared__ float l_tile;
    __shared__ float scores[B];
    __shared__ float p[B];

    // ########################################################

    int bh_idx=blockIdx.x;
    int tid=threadIdx.x;

    // Q has shape (batch*heads, 1, head_dim), so offset is bh_idx * head_dim
    const float* q_bh = q + bh_idx * head_dim;
    const float* k_bh = k + bh_idx * seq_len * head_dim;
    const float* v_bh = v + bh_idx * seq_len * head_dim;
    float* o_bh = out + bh_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for(int i=tid;i<head_dim;i+=blockDim.x){
        q_tile[i]=q_bh[i];
        o_tile[i]=0.0f;
    }
    if (tid==0){
        m_tile=-FLT_MAX;
        l_tile=0.0f;
    }
    __syncthreads();

    int num_kv_blocks=(seq_len+B-1)/B;
    for (int kv_block_idx=0;kv_block_idx<num_kv_blocks;kv_block_idx++){
        int kv_start=kv_block_idx*B;

        if(tid<B){
            int kv_row=kv_start + tid;
            if (kv_row<seq_len){
                for (int d=0;d<head_dim;d++){
                    k_tile[tid][d] = k_bh[kv_row * head_dim + d];
                    v_tile[tid][d] = v_bh[kv_row * head_dim + d];
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    k_tile[tid][d] = 0.0f;
                    v_tile[tid][d] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        if (tid<B){
            int kv_col=kv_start+tid;
            float score=0.0f;
            if (kv_col<seq_len){
                for (int d=0;d<head_dim;d++){
                    score+=q_tile[d] * k_tile[tid][d];
                }
                score*=scale;
            }
            else{
                score=-FLT_MAX;
            }
            scores[tid]=score;
        }
        __syncthreads();

        if (tid==0){
            float m_curr=-FLT_MAX;
            for (int j=0;j<B;j++){
                if (scores[j] > m_curr) {
                    m_curr = scores[j];
                }
            }
            float m_new=fmaxf(m_tile, m_curr);
            float l_row=0.0f;
            for (int j = 0; j < B; j++) {
                float pval = expf(scores[j] - m_new);
                p[j] = pval;
                l_row += pval;
            }
            float alpha=expf(m_tile-m_new); //correction factor
            float l_new=alpha * l_tile + l_row;
            for (int d = 0; d < head_dim; d++) {
                float o_val = alpha * o_tile[d];
                for (int j = 0; j < B; j++) {
                    o_val += p[j] * v_tile[j][d];
                }
                o_tile[d] = o_val;
            }
            
            // Update m and l
            m_tile = m_new;
            l_tile = l_new;
        }
        __syncthreads();
    }

    if (tid == 0) {
            float l_inv = 1.0f / l_tile;
            for (int d = 0; d < head_dim; d++) {
                o_bh[d] = o_tile[d] * l_inv;
            }
    }


            



    // ########################################################
}

torch::Tensor custom_flash_attention_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, int num_heads, bool causal) {
    auto options = q.options();
    
    int batch_size = q.size(0);
      
    int seq_len = k.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    
    // Reshape to (batch_size * num_heads, 1, head_dim)
    auto q_bh = q.view({batch_size, 1, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, 1, head_dim});
    auto k_bh = k.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    auto v_bh = v.view({batch_size, seq_len, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size * num_heads, seq_len, head_dim});
    
    auto out_bh = torch::zeros_like(q_bh, options);
    
    // ########################################################

    const int B = 32;  // KV tile size
    int total_batch_heads = batch_size * num_heads;
    
    dim3 grid(total_batch_heads);
    dim3 block(B);  // B threads per block
    
    flash_attention_decode_kernel<32><<<grid, block>>>(
        q_bh.data_ptr<float>(),
        k_bh.data_ptr<float>(),
        v_bh.data_ptr<float>(),
        out_bh.data_ptr<float>(),
        seq_len,
        head_dim,
        causal
    );
    
    // ########################################################

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape back to (batch_size, 1, hidden_dim)
    auto out = out_bh.view({batch_size, num_heads, 1, head_dim})
                     .permute({0, 2, 1, 3})
                     .contiguous()
                     .view({batch_size, 1, hidden_dim});
    
    return out;
}

__global__ void update_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int batch_size,
    int current_pos,
    int max_seq_len,
    int hidden_dim
) {
    // ########################################################
    // Each thread copies one element (one batch, one hidden_dim index)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_dim;
    if (idx >= total_elements) return;
    //2 requests means 2 batches
    int b=idx/hidden_dim;
    int d=idx%hidden_dim;
    // Source: k/v have shape (batch_size, 1, hidden_dim)
    int src_idx=b * hidden_dim + d;
    // Destination: k_cache/v_cache have shape (batch_size, max_seq_len, hidden_dim)
    // k_cache[b, current_pos, d] = k_cache[b * max_seq_len * hidden_dim + current_pos * hidden_dim + d]
    int dst_idx = b * max_seq_len * hidden_dim + current_pos * hidden_dim + d;
    k_cache[dst_idx] = k[src_idx];
    v_cache[dst_idx] = v[src_idx];

    // ########################################################
}

void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k, torch::Tensor v, int current_pos) {
    auto options = k_cache.options();

    int batch_size = k_cache.size(0);
    int max_seq_len = k_cache.size(1);
    int hidden_dim = k_cache.size(2);
    
    int total_threads = batch_size * hidden_dim;

    int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks((total_threads + threads_per_block - 1) / threads_per_block);

    update_cache_kernel<<<blocks, threads>>>(
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        batch_size,
        current_pos,
        max_seq_len,
        hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_flash_attention_decode", &custom_flash_attention_decode, "Custom FlashAttention in CUDA");
    m.def("update_kv_cache", &update_kv_cache, "Update KV cache in CUDA");
}
