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
    __shared__ float q_tile[B_r][128];  // Max head_dim = 128
    __shared__ float k_tile[B_c][128];
    __shared__ float v_tile[B_c][128];
    __shared__ float o_tile[B_r][128];
    __shared__ float m_tile[B_r];
    __shared__ float l_tile[B_r];
    __shared__ float scores[B_r][B_c];
    __shared__ float p[B_r][B_c];

    int bh_idx=blockIdx.x; //as the first dimension of grid is batch_size * num_heads
    int q_block_idx=blockIdx.y; //as the second dimension of grid is number of q blocks
    int tid=threadIdx.x;

    int q_start=q_block_idx*B_r;
    if (q_start>=seq_len){
        return;
    }

    const float* q_bh=q + bh_idx * seq_len * head_dim;
    const float* k_bh=k + bh_idx * seq_len * head_dim;
    const float* v_bh=v + bh_idx * seq_len * head_dim;
    float* o_bh=out + bh_idx * seq_len * head_dim;
    float scale=1.0f / sqrtf((float)head_dim);
    //each thread handles one row in the q block
    int local_row=tid; // row insiex in current q_tile
    int global_row=q_start+local_row;// row index in current batch-head
    bool row_valid = (global_row < seq_len) && (local_row < B_r);

    if (row_valid){
        for (int d=0;d<head_dim;d++){
            q_tile[local_row][d]=*(q_bh+global_row*head_dim+d);
        }
    } else if (local_row<B_r){ //padding for last q block
        for (int d=0;d<head_dim;d++){
            q_tile[local_row][d]=0.0f;
        }
    }
   
    if (local_row < B_r) { //init
        m_tile[local_row] = -FLT_MAX;
        l_tile[local_row] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            o_tile[local_row][d] = 0.0f;
        }
    }
    __syncthreads();

    int num_kv_blocks=(seq_len+B_c-1)/B_c;

    for (int kv_block_idx=0;kv_block_idx<num_kv_blocks;kv_block_idx++){
        int kv_start=kv_block_idx*B_c;

        if (causal && kv_start > q_start+B_r-1){
            break;
        }

        if (local_row < B_c){//load k and  tile
            int kv_row=kv_start + local_row;
            if (kv_row<seq_len){
                for (int d=0;d<head_dim;d++){
                    k_tile[local_row][d] = k_bh[kv_row * head_dim + d];
                    v_tile[local_row][d] = v_bh[kv_row * head_dim + d];
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    k_tile[local_row][d] = 0.0f;
                    v_tile[local_row][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        //attnetion computation begins 

        if (local_row < B_r){
            for (int j=0;j<B_c;j++){
                int kv_col=kv_start+j;
                float score=0.0f;

                if (global_row < seq_len && kv_col < seq_len){
                    for (int d=0;d<head_dim;d++){
                        score+=q_tile[local_row][d] * k_tile[j][d];
                    }
                    score*=scale;
                
                    if (causal && kv_col > global_row){
                        score=-FLT_MAX;
                    }
                }else{
                    score=-FLT_MAX;
                }
                scores[local_row][j]=score;
            }
        }
        __syncthreads();

        if (row_valid){
            float m_row=-FLT_MAX;
            for (int j=0;j<B_c;j++){
                if (scores[local_row][j] > m_row) {
                    m_row = scores[local_row][j];
                }
            }
            float m_new=fmaxf(m_tile[local_row], m_row);// max updated
            // Compute P = exp(S - m_new) and sum
            float l_row=0.0f;
            for (int j = 0; j < B_c; j++) {
                float pval = expf(scores[local_row][j] - m_new);
                p[local_row][j] = pval;
                l_row += pval;
            }
            
            float alpha=expf(m_tile[local_row]-m_new); //correction factor
            float l_new=alpha * l_tile[local_row] + l_row;
            // Update output: O = alpha * O + P @ V
            for (int d = 0; d < head_dim; d++) {
                float o_val = alpha * o_tile[local_row][d];
                for (int j = 0; j < B_c; j++) {
                    o_val += p[local_row][j] * v_tile[j][d];
                }
                o_tile[local_row][d] = o_val;
            }
            
            // Update m and l
            m_tile[local_row] = m_new;
            l_tile[local_row] = l_new;
        }
        __syncthreads();
    }
    // Normalize output and write to global memory
    if (row_valid){
        float l_inv=1.0f / l_tile[local_row];
        for (int d = 0; d < head_dim; d++) {
            o_bh[global_row * head_dim + d] = o_tile[local_row][d] * l_inv;
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
    // const int B_c = 16;  // KV tile size (columns)
    
    int num_q_blocks = (seq_len + B_r - 1) / B_r;
    int total_batch_heads = batch_size * num_heads;
    
    // Grid: (batch_size * num_heads, num_q_blocks)
    // Block: B_r threads (one thread per query row in the tile)
    dim3 grid(total_batch_heads, num_q_blocks);
    dim3 block(B_r);
    
    flashattention_kernel<16, 16><<<grid, block>>>(
        q_bh.data_ptr<float>(),
        k_bh.data_ptr<float>(),
        v_bh.data_ptr<float>(),
        out_bh.data_ptr<float>(),
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
