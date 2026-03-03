#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <cmath>

using namespace torch::indexing;

__global__ void softmax_kernel_batched(float *inp, float *outp, int NUM_ROW, int NUM_COL)
{
    extern __shared__ float buffer[];

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (row >= NUM_ROW) return;

    int offset = batch * NUM_ROW * NUM_COL + row * NUM_COL;
    float *inp_row = inp + offset;
    float *out_row = outp + offset;

    float thread_row_max = -FLT_MAX;

    for(int i = tid; i < NUM_COL; i += num_threads){
        thread_row_max = std::fmax(thread_row_max, inp_row[i]);
    }

    buffer[tid] = thread_row_max;

    // do a reduction to find the rowmax across the threads
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            buffer[tid] = std::fmaxf(buffer[tid], buffer[tid + stride]);
        }
    }
    
    __syncthreads();
    float row_max = buffer[0];

    float thread_row_sum = 0.0f;

    for (int i = tid; i < NUM_COL; i += num_threads) {
        float val = std::exp(inp_row[i] - row_max);
        out_row[i] = val;
        thread_row_sum += val;
    }

    buffer[tid] = thread_row_sum;

    // do a reduction to find the rowsum across the threads
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            buffer[tid] += buffer[tid + stride];
        }
    }

    __syncthreads();
    float row_sum = buffer[0];

    for (int i = tid; i < NUM_COL; i += num_threads) {
        out_row[i] /= row_sum;
    }
}

template <int TILE_SIZE>
__global__ void GEMM_NT_kernel_batched(
    float *a_mat, float *b_mat, float *out_mat,
    int M, int N, int K
) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for(int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++){
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            // a_tile[ty][k] = A[row, k]
            a_tile[ty][tx] = a_mat[(b * M * K) + (row * K) + a_col];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        int b_col = t * TILE_SIZE + ty;
        if (col < N && b_col < K) {
            // b_tile[k][tx] = B[col, k]
            b_tile[ty][tx] = b_mat[(b * K * N) + (col * K) + b_col];
        } else {
            b_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++){
            sum += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        out_mat[(b * M * N) + (row * N) + col] = sum;
    }
}

template <int TILE_SIZE>
__global__ void GEMM_NN_kernel_batched(
    float *a_mat, float *b_mat, float *out_mat,
    int M, int N, int K
) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++){
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            // a_tile[ty][k] = A[row, k]
            a_tile[ty][tx] = a_mat[(b * M * K) + (row * K) + a_col];
        } else {
            a_tile[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            // b_tile[k][tx] = B[k, col]
            b_tile[ty][tx] = b_mat[(b * K * N) + (b_row * N) + col];
        } else {
            b_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++){
            sum += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        out_mat[(b * M * N) + (row * N) + col] = sum;
    }
}

__global__ void scale_and_causal_mask_batched(
    float *mat, int num_rows, int num_cols,
    float scale, bool causal
) {
    int b = blockIdx.x;
    int row_id = blockIdx.y;
    int tid = threadIdx.x;

    if (row_id >= num_rows) return;

    float *row = mat + (b * num_rows * num_cols) + (row_id * num_cols);

    for (int j = tid; j < num_cols; j += blockDim.x) {
        row[j] = (causal && j > row_id)
                  ? -FLT_MAX
                  : row[j] * scale;
    }
}

torch::Tensor custom_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    int num_heads, bool causal
) {
    auto options = q.options();

    const int TILE_SIZE = 16;

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;

    // Reshape  (batch_size * num_heads, seq_len, head_dim) 
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

    auto qk = torch::empty({batch_size * num_heads, seq_len, seq_len}, options);
    auto s = torch::empty_like(qk);
    auto o_bh = torch::empty({batch_size * num_heads, seq_len, head_dim}, options);

    // Q @ K.mT
    int BH = batch_size * num_heads;
    int M = seq_len;
    int N = seq_len;
    int K = head_dim;

    dim3 block_size_gemm_nt(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_gemm_nt((N + TILE_SIZE - 1) / TILE_SIZE,
                           (M + TILE_SIZE - 1) / TILE_SIZE,
                           BH);

    GEMM_NT_kernel_batched<TILE_SIZE>
        <<<grid_size_gemm_nt, block_size_gemm_nt>>>(
        q_bh.data_ptr<float>(), k_bh.data_ptr<float>(), qk.data_ptr<float>(),
        M, N, K);
                
    // Scale + causal mask (batched over BH, rows)
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 threads_per_block_scale(256);
    dim3 blocks_per_grid_scale(BH, seq_len);

    scale_and_causal_mask_batched
        <<<blocks_per_grid_scale, threads_per_block_scale>>>(
        qk.data_ptr<float>(), seq_len, seq_len, scale, causal);

    // Softmax (batched along last dim)
    dim3 block_size_softmax(256);
    dim3 grid_size_softmax(BH, seq_len);
    size_t shm_size_softmax = block_size_softmax.x * sizeof(float);

    softmax_kernel_batched
        <<<grid_size_softmax, block_size_softmax, shm_size_softmax>>>(
        qk.data_ptr<float>(), s.data_ptr<float>(), seq_len, seq_len);
    
    // S @ V (batched)
    int M_nn = seq_len;
    int N_nn = head_dim;
    int K_nn = seq_len;

    dim3 block_size_gemm_nn(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_gemm_nn((N_nn + TILE_SIZE - 1) / TILE_SIZE,
                           (M_nn + TILE_SIZE - 1) / TILE_SIZE,
                           BH);

    GEMM_NN_kernel_batched<TILE_SIZE>
        <<<grid_size_gemm_nn, block_size_gemm_nn>>>(
        s.data_ptr<float>(), v_bh.data_ptr<float>(), o_bh.data_ptr<float>(),
        M_nn, N_nn, K_nn);

    // Reshape back (batch_size, seq_len, hidden_dim)
    auto o = o_bh.view({batch_size, num_heads, seq_len, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous()
                 .view({batch_size, seq_len, hidden_dim});

    return o;
}

torch::Tensor test_batched_softmax(torch::Tensor inp, int dim) {
    int N_DIM = inp.dim();
    if (dim < 0) {
        dim += N_DIM;
    }
    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
    }
    // Ensure contiguous memory layout for linear indexing in the CUDA kernel
    inp = inp.contiguous();

    auto inp_sizes = inp.sizes();
    int N_BATCH = inp_sizes[0];
    int N_ROW = inp_sizes[1];
    int N_COL = inp_sizes[2];
    dim3 threads_per_block(256);
    dim3 blocks_per_grid(N_BATCH, N_ROW);
    auto outp = torch::empty(inp_sizes, inp.options());
    
    //  shared memory for each thread in the block
    softmax_kernel_batched<<<blocks_per_grid, threads_per_block, threads_per_block.x * sizeof(float)>>>(
        inp.data_ptr<float>(), outp.data_ptr<float>(), N_ROW, N_COL);
    
    if (dim != N_DIM - 1) {
        inp = inp.transpose(dim, N_DIM - 1);
        outp = outp.transpose(dim, N_DIM - 1);
    }

    return outp;
}

torch::Tensor test_batched_GEMM_NN(torch::Tensor a, torch::Tensor b)
{
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[2], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NN_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

torch::Tensor test_batched_GEMM_NT(torch::Tensor a, torch::Tensor b)
{
    auto a_size = a.sizes();
    auto b_size = b.sizes();
    int N_BATCH = a_size[0];
    int M = a_size[1], N = b_size[1], K = a_size[2];

    auto out = torch::empty({N_BATCH, M, N}, a.options());
    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        N_BATCH);
    GEMM_NT_kernel_batched<TILE_SIZE><<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_batched_softmax", &test_batched_softmax, "Testing interface for custom softmax in CUDA");
    m.def("test_batched_GEMM_NN", &test_batched_GEMM_NN, "Testing interface for custom matmul (A @ B) in CUDA");
    m.def("test_batched_GEMM_NT", &test_batched_GEMM_NT, "Testing interface for custom matmul (A @ B^T) in CUDA");
    m.def("custom_attention", &custom_attention, "Custom attention in CUDA");
}