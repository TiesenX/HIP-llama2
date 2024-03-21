#pragma once
#include "build_schedule.h"

// Macros for error checking
#define CHECK_HIP(cmd)                                                                   \
  do {                                                                                   \
    hipError_t error = (cmd);                                                            \
    if (error != hipSuccess)                                                             \
    {                                                                                    \
      std::cerr << "HIP error (" << hipGetErrorString(error) << ") at line "             \
                << __LINE__ << " in file " << __FILE__ << "\n";                          \
      exit(-1);                                                                          \
    }                                                                                    \
  } while (0)

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset >>= 1) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float warpReduceMax(float val) {
  for (int offset = warpSize/2; offset > 0; offset >>= 1) 
    val = max(val, __shfl_down(val, offset));
  return val;
}

#define WARP_SIZE 64
__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ float shared[WARP_SIZE]; // Shared mem for 16 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__inline__ __device__
float blockReduceMax(float val) {

  static __shared__ float shared[WARP_SIZE]; // Shared mem for 16 partial max values
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceMax(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

  return val;
}

// Utility routine to divide a into ceiling of b parts
int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

const int num_threads_lrg = 1024;
const int num_threads_med = 256;

__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size) {
  // parallel reduction of sum of squares via CUB
  const int tid = threadIdx.x;
  const int idx = blockIdx.x; // batch index
  const int num_threads = blockDim.x;

  float ss = 0.0f;
  for (int i = tid; i < size; i+=num_threads) {
      ss += x[i + idx * size] * x[i + idx * size];
  }
  ss = blockReduceSum(ss);

  // serialization point to calculate normalization factor 
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
      ss /= size;
      ss += 1e-5f;
      ss = 1.0f / sqrtf(ss);
      shared_ss = ss;
  }
  __syncthreads();
  ss = shared_ss;

  // normalize and scale
  for (int i = tid; i < size; i+=num_threads) {
      o[i + idx * size] = weight[i] * (ss * x[i + idx * size]);
  }
}

void gpu_rmsnorm(float* o, float* x, float* weight, int size, int batch_size) {
  // for (int idx = 0; idx < batch_size; idx++)
  rmsnorm_kernel <<<batch_size, num_threads_lrg>>> (o, x, weight, size);
}


__device__ float blockReduceMultipleSum(float val)
{
    static __shared__ float shared[WARP_SIZE]; // Each warp write its sum reduction at here (AMD only have max 16 warp but I allocated more for convenience)

    int blockId = threadIdx.y * blockDim.x + threadIdx.x; // block index in 2d block
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int lane = blockId % WARP_SIZE;
    int wid = blockId / WARP_SIZE;
    int batch = threadIdx.y;
    int batchNumber = blockDim.y;
    int warpNumber = (blockSize + WARP_SIZE - 1) / WARP_SIZE;
    int warpEachBatch = (warpNumber + batchNumber - 1) / batchNumber;

    val = warpReduceSum(val); // Each warp perform sum reduction

    if (lane == 0)
        shared[wid] = val; // First thread in each warp write warp sum to shared mem
    __syncthreads();

    float batchReduction = 0.0f;
    for (int w = 0; w < warpEachBatch; w++)
    {
        int warpInBatch = batch * warpEachBatch + w;
        if (warpInBatch < warpNumber)
            batchReduction += shared[warpInBatch];
    }

    // return results[batch];
    return batchReduction;
}

__global__ void matmulGPU(float *xout, float *x, float *w, int N, int D)
{
    int tid = threadIdx.x;
    int batch = threadIdx.y;
    int globalD = blockIdx.x;
    int threadNumInBlock = blockDim.x;

    // Move pointer
    xout += batch * D;
    x += batch * N;

    float val = 0.0f;
    for (int el = tid; el < N; el += threadNumInBlock)
    {
        val += w[el + globalD * N] * x[el];
    }

    // Return based on batch of this val
    val = blockReduceMultipleSum(val); // return reduction result based on batch

    if (tid == 0) // thread 0 in each batch write result to
    {
        xout[globalD] = val;
    }
}

void gpu_matmul(float *xout, float *x, float *w, int N, int D, int batch_size)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function

    dim3 threadBlock(128, batch_size);
    dim3 gridSize(D);
    matmulGPU<<<gridSize, threadBlock>>>(xout, x, w, N, D);
    CHECK_HIP(hipGetLastError());
}

__global__ void mat_vec(float *xout, float *x, float *w, int n, int d) {

  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int d_i = blockIdx.x;
  float val = 0.0f;

  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    val += w[d_i * n + idx] * x[idx];
  }
  val = blockReduceSum(val);

  if (threadIdx.x == 0) {
    xout[d_i] = val;
  }
}

void gpu_mat_vec(float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function

    dim3 threadBlock(1024);
    dim3 gridSize(d);
    mat_vec<<<gridSize, threadBlock>>>(xout, x, w, n, d);
}

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    __shared__ float shared_val;
    max_val = blockReduceMax(max_val);
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

// void gpu_matmul(float* xout, float* x, float* w, int n, int d, int batch_size) {
//   dim3 grid(d);
//   // printf("batch_size: %d\n", batch_size);
//   matmul_kernel<<<grid, 1024>>>(xout, x, w, n, d, batch_size);
//     // CHECK_HIP(hipGetLastError());
// }

// #define TILE_WIDTH 16
// __global__ void matmul_kernel(float *C, const float *A, const float *B, int n, int m, int k)
// {
//   // A (k, n) @ B (m,n) -> C (k, m)
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int k_id = by * blockDim.y + ty;
//   int m_id = bx * blockDim.x + tx;
//   float sum = 0.0;
//   __shared__ float As[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
//   A += k_id * n + tx;
//   B += m_id * n + ty;
//   for (int n_offset = 0; n_offset < n; n_offset += TILE_WIDTH)
//   {
//     As[ty][tx] = (k_id < k && n_offset + tx < n) ? A[n_offset] : 0.0;
//     Bs[ty][tx] = (m_id < m && n_offset + ty < n) ? B[n_offset] : 0.0;
//     __syncthreads();
//     for (int e = 0; e < TILE_WIDTH; ++e)
//       sum += As[ty][e] * Bs[e][tx];
//     __syncthreads();
//   }
//   if (k_id < k && m_id < m)
//     C[k_id * m + m_id] = sum;
// }

// void gpu_matmul(float *C, float *A, float *B, int n, int m, int k)
// {
//   // A (k, n) @ B (m,n) -> C (k, m)
//   dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
//   dim3 gridDim((m + TILE_WIDTH - 1) / TILE_WIDTH, (k + TILE_WIDTH - 1) / TILE_WIDTH);
//   matmul_kernel<<<gridDim, blockDim>>>(C, A, B, n, m, k);
//   CHECK_HIP(hipGetLastError());
// }

// #define TILE_WIDTH 16

// __global__ void matmul_kernel(float *C, const float *A, const float *B, int n, int m, int k)
// {
//     // Shared memory for tiles
//     __shared__ float As[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

//     // Calculate thread and block indices
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     // Calculate global indices
//     int k_id = by * blockDim.y + ty;
//     int m_id = bx * blockDim.x + tx;

//     // Initialize the sum
//     float sum = 0.0;

//     // Loop over tiles
//     for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
//     {
//         // Load tiles into shared memory
//         int col = t * TILE_WIDTH + tx;
//         int row = t * TILE_WIDTH + ty;

//         if (row < n && k_id < k)
//             As[ty][tx] = A[k_id * n + col];
//         else
//             As[ty][tx] = 0.0;

//         if (col < n && m_id < m)
//             Bs[ty][tx] = B[m_id * n + row];
//         else
//             Bs[ty][tx] = 0.0;

//         __syncthreads();

//         // Compute partial sum for the tile
//         for (int i = 0; i < TILE_WIDTH; ++i)
//             sum += As[ty][i] * Bs[i][tx];

//         __syncthreads();
//     }

//     // Store the result to global memory
//     if (k_id < k && m_id < m)
//         C[k_id * m + m_id] = sum;
// }

// void gpu_matmul(float *C, float *A, float *B, int n, int m, int k)
// {
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
//     dim3 gridDim((m + TILE_WIDTH - 1) / TILE_WIDTH, (k + TILE_WIDTH - 1) / TILE_WIDTH);
//     matmul_kernel<<<gridDim, blockDim>>>(C, A, B, n, m, k);
//     CHECK_HIP(hipGetLastError());
// }

void softmax(float* x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}


void matmul(float* xout, float* x, float* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
  #pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

__global__ void RoPE_kernel(int pos, float* sq, float* sk, 
                            int dim, int kv_dim, int head_size) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (i >= dim) return;

  int head_dim = i % head_size;
  float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
  float val = pos * freq;
  float fcr = cosf(val);
  float fci = sinf(val);
  int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
  for (int v = 0; v < rotn; v++) {
    float* vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
    float v0 = vec[i];
    float v1 = vec[i+1];
    vec[i]   = v0 * fcr - v1 * fci;
    vec[i+1] = v0 * fci + v1 * fcr;
  }
}
void gpu_RoPE(float* sq, float* sk, int* pos, int dim, int head_size, int kv_dim, int* new_batch_slot, int curr_batch_size) {
  dim3 block(64);
  dim3 grid(((dim + 1) / 2 + block.x - 1) / block.x);

  for (int idx = 0; idx < curr_batch_size; idx++) {
    RoPE_kernel<<<grid, block>>>(pos[idx], sq + idx * dim, 
                                           sk + pos[idx] * BATCH_SIZE * kv_dim + new_batch_slot[idx] * kv_dim, 
                                           dim, kv_dim, head_size);
  }
}


void RoPE(float* sq, float* sk, int pos, int dim, int head_size, int kv_dim) { //s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

#define MAX_SEQ_LEN 8192
__global__ void MultiHeadAttention_kernel(float* __restrict__ output, const float* __restrict__ sq,
    const float* __restrict__ key_cache, const float* __restrict__ value_cache,
    int kv_dim, int kv_mul, int head_size, int loff, int seq_len, int batch, int batch_size) {

    int h = blockIdx.x;

    // get the query vector for this head
    const float* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        // const float* k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        const float* k = key_cache + loff + t * batch_size * kv_dim + batch * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;

        for (int i = 0; i < head_size; i++)
            score += q[i] * k[i];
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();

    // softmax the scores to get attention weights
    softmax_gpu(att, seq_len);
    __syncthreads();

    // weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        
        #pragma unroll
        for (int t = 0; t < seq_len; t++)
            // val += att[t] * value_cache[loff + t * kv_dim + (h / kv_mul) * head_size + i];
            val += att[t] * value_cache[loff + t * batch_size * kv_dim + batch * kv_dim + (h / kv_mul) * head_size + i];
              // float *v = s_valuecache + loff + t * batch_size * kv_dim + batch * kv_dim + (h / kv_mul) * head_size;
        output[(h / kv_mul) * head_size + i] = val;
    }
}

void gpu_MultiHeadAttention(float *output, float *q, float *key_cache, float *value_cache, 
                            int kv_dim, int kv_mul, int num_heads, int head_size, int loff, int seq_len, int batch, int batch_size, hipStream_t* stream) {
    MultiHeadAttention_kernel<<<num_heads, num_threads_lrg, 0, *stream>>>(output, q, key_cache, value_cache, 
                                                              kv_dim, kv_mul, head_size, loff, seq_len, batch, batch_size);
}

void MultiHeadAttention(int pos, Config* p, RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
  int h;
  #pragma omp parallel for private(h)
  for (h = 0; h < p->n_heads; h++) {
    // get the query vector for this head
    float* q = s->q + h * head_size;
    // attention scores for this head
    float* att = s->att + h * p->seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // calculate the attention score as the dot product of q and k
      float score = 0.0f;
      for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i];
      }
      score /= sqrtf(head_size);
      // save the score to the attention buffer
      att[t] = score;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    float* xb = s->xb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      // accumulate the weighted value into xb
      for (int i = 0; i < head_size; i++) {
          xb[i] += a * v[i];
      }
    }
  }
}

__global__ void swiglu_kernel(float *shb, float *shb2, int hidden_dim, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < hidden_dim * batch_size; i += stride) {
        int h_index = i % hidden_dim; // Get index within one hidden_dim
        int batch_index = i / hidden_dim; // Get batch index

        float val = shb[i];
        // Apply Swish activation function
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val *= sigmoid_val;

        // Element-wise multiply with shb2
        val *= shb2[i];
        shb[i] = val;
    }
}

void gpu_swiglu(float *shb, float *shb2, int hidden_dim, int batch_size) {
    int threads_per_block = 256;
    int num_blocks = (hidden_dim * batch_size + threads_per_block - 1) / threads_per_block;
    swiglu_kernel<<<num_blocks, threads_per_block>>>(shb, shb2, hidden_dim, batch_size);
}

void swiglu(float *shb, float *shb2, int hidden_dim) {
    for (int i = 0; i < hidden_dim; i++) {
        float val = shb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= shb2[i];
        shb[i] = val;
    }
}

__global__ void accum_kernel(float* a, const float* b, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size * batch_size; i += stride) {
        a[i] += b[i];
    }
}

void gpu_accum(float *a, const float *b, int size, int batch_size) {
    int threads_per_block = 256;
    int num_blocks = (size * batch_size + threads_per_block - 1) / threads_per_block;
    accum_kernel<<<num_blocks, threads_per_block>>>(a, b, size, batch_size);
}

void accum(float *a, float *b, int size) {
  for (int i = 0; i < size; i++) {
    a[i] += b[i];
  }
}