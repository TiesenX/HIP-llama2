/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <fstream>
#include <iostream>

#include <hip/hip_runtime.h>
#include <omp.h>
#include <math.h>
#include "run.h"

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
// Transformer model


// void malloc_run_state(RunState* s, Config* p) {
//   // we calloc instead of malloc to keep valgrind happy
//   int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//   s->x = (float*)calloc(p->dim, sizeof(float));
//   s->xb = (float*)calloc(p->dim, sizeof(float));
//   s->xb2 = (float*)calloc(p->dim, sizeof(float));
//   s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
//   s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
//   s->q = (float*)calloc(p->dim, sizeof(float));
//   s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
//   s->logits = (float*)calloc(p->vocab_size, sizeof(float));
//   // ensure all mallocs went fine
//   if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
//       || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
//     fprintf(stderr, "malloc failed!\n");
//     exit(EXIT_FAILURE);
//   }
// }

#ifdef USE_GPU
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CHECK_HIP(hipMalloc((void**)&s->x, p->dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->xb, p->dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->xb2, p->dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->hb, p->hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->q, p->dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float)));
    CHECK_HIP(hipHostMalloc((void**)&s->logits, p->vocab_size * sizeof(float), hipMemAllocationTypePinned));
    // s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    CHECK_HIP(hipFree(s->x));
    CHECK_HIP(hipFree(s->xb));
    CHECK_HIP(hipFree(s->xb2));
    CHECK_HIP(hipFree(s->hb));
    CHECK_HIP(hipFree(s->hb2));
    CHECK_HIP(hipFree(s->q));
    CHECK_HIP(hipFree(s->att));
    CHECK_HIP(hipFree(s->logits_gpu));
    CHECK_HIP(hipHostFree(s->logits));    
    // free(s->logits);
    CHECK_HIP(hipFree(s->key_cache));
    CHECK_HIP(hipFree(s->value_cache));
}

#elif KERNEL_TEST
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CHECK_HIP(hipHostMalloc((void**)&s->x, p->dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->xb, p->dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->xb2, p->dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->hb, p->hidden_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->q, p->dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&s->logits, p->vocab_size * sizeof(float), hipMemAllocationTypePinned));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    CHECK_HIP(hipHostFree(s->x));
    CHECK_HIP(hipHostFree(s->xb));
    CHECK_HIP(hipHostFree(s->xb2));
    CHECK_HIP(hipHostFree(s->hb));
    CHECK_HIP(hipHostFree(s->hb2));
    CHECK_HIP(hipHostFree(s->q));
    CHECK_HIP(hipHostFree(s->att));
    CHECK_HIP(hipHostFree(s->logits_gpu));
    CHECK_HIP(hipHostFree(s->logits));    
    CHECK_HIP(hipHostFree(s->key_cache));
    CHECK_HIP(hipHostFree(s->value_cache));
}

#else
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}
#endif

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight = ptr;
  ptr += n_layers * p->dim;
  w->wq = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight = ptr;
  ptr += n_layers * p->dim;
  w->w1 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls = shared_weights ? w->token_embedding_table : ptr;
}


void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
#ifdef USE_GPU
    // allocate & copy mmap data to the gpu first
    // TODO: allocate & copy just a portion to the GPU if the weights are too big
    // to fit in the GPU, then copy the data only as needed while running.
    float* weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CHECK_HIP(hipMalloc((void**)&weights_ptr, weights_size));
    CHECK_HIP(hipMemcpy(weights_ptr, *data + sizeof(Config)/sizeof(float), weights_size, hipMemcpyHostToDevice));
#elif KERNEL_TEST
    float* weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CHECK_HIP(hipHostMalloc((void**)&weights_ptr, weights_size, hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(weights_ptr, *data + sizeof(Config)/sizeof(float), weights_size, hipMemcpyHostToDevice));
#else
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
#endif
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void print_transformer(Transformer* t) {
  printf("---------Model Information----------\n");
  printf("dim: %d\n", t->config.dim);
  printf("hidden_dim: %d\n", t->config.hidden_dim);
  printf("n_layers: %d\n", t->config.n_layers);
  printf("n_heads: %d\n", t->config.n_heads);
  printf("n_kv_heads: %d\n", t->config.n_kv_heads);
  printf("vocab_size: %d\n", t->config.vocab_size);
  printf("seq_len: %d\n", t->config.seq_len);
  printf("weights_size: %lu MB\n", (t->file_size - sizeof(Config)) / (1024L*1024L));
  printf("------------------------------------\n");
}


void build_transformer(Transformer *t, char* checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
  print_transformer(t);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
#ifdef USE_GPU
    // we hipMalloc a region of memory, then hand the address to
    // the token_embedding_table field.  Free it here.
    CHECK_HIP(hipFree(t->weights.token_embedding_table));
#elif KERNEL_TEST
    CHECK_HIP(hipHostFree(t->weights.token_embedding_table));
#endif
    // free the RunState buffers
    free_run_state(&t->state);
}
// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float warpReduceMax(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val = max(val, __shfl_down(val, offset));
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ float shared[16]; // Shared mem for 16 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__inline__ __device__
float blockReduceMax(float val) {

  static __shared__ float shared[16]; // Shared mem for 16 partial max values
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceMax(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

  return val;
}

// Utility routine to divide a into ceiling of b parts
int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

const int num_threads_lrg = 1024;
const int num_threads_med = 256;

__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size, int elementsPerThread) {
    // parallel reduction of sum of squares via CUB
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;
        if (j < size)
            ss += x[j] * x[j];
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
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}
void gpu_rmsnorm(float* o, float* x, float* weight, int size) {
    int elementsPerThread = divUp(size, num_threads_lrg);
    rmsnorm_kernel <<<1, num_threads_lrg >>> (o, x, weight, size, elementsPerThread);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
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

__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {

  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i = blockIdx.x * blockDim.x + threadIdx.x;
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

void gpu_matmul(float* xout, float* x, float* w, int n, int d) {
  matmul_kernel<<<d, 512>>>(xout, x, w, n, d);
  CHECK_HIP(hipGetLastError());
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
void gpu_RoPE(float* sq, float* sk, int pos, int dim, int head_size, int kv_dim) {
  dim3 block(64);
  dim3 grid(((dim + 1) / 2 + block.x - 1) / block.x);
  RoPE_kernel<<<grid, block>>>(pos, sq, sk, dim, kv_dim, head_size);
  CHECK_HIP(hipGetLastError());
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
// __global__ void MultiHeadAttention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache, float *value_cache, int kv_dim, int kv_mul, int head_size, int loff, int dim) {
  // int h = blockIdx.x;
  // // get the query vector for this head
  // float* q = sq + h * head_size;
  // // attention scores for this head
  // float* att = satt + h * seq_len;
  // // iterate over all timesteps, including the current one 
  // // In hip, each thread does a small portion of the calc
  // for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
  //   // get the key vector for this head and at this timestep
  //   float* k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
  //   // calculate the attention score as the dot product of q and k
  //   float score = 0.0f;
  //   for (int i = 0; i < head_size; i++) {
  //       score += q[i] * k[i];
  //   }
  //   score /= sqrtf(head_size);
  //   // save the score to the attention buffer
  //   att[t] = score;
  // }
  // // above was this threads portion of the iteration.  wait for all threads to finish
  // __syncthreads();

  // // softmax the scores to get attention weights, from 0..pos inclusively
  // softmax_gpu(att, pos + 1);
  // __syncthreads();

  // // weighted sum of the values, store back into xb
  // // NOTE: by swapping the order of the for loops (vs. C) a simpler
  // // version of the code accomplishes the same task and fits more
  // // naturally with the hip way of subdividing the problem.
  // float* xb = sxb + h * head_size;
  // for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
  //   float val = 0.0f;
  //   for (int t = 0; t <= pos; t++) {
  //       // get the value vector for this head and at this timestep
  //       float* v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
  //       // get the attention weight for this timestep
  //       float a = att[t];
  //       val += a * v[i];
  //   }
  //   xb[i] = val;
  // }
__global__ void MultiHeadAttention_kernel(float* __restrict__ output, const float* __restrict__ sq,
    const float* __restrict__ key_cache, const float* __restrict__ value_cache,
    int num_heads, int head_size, int loff, int seq_len, int dim) {

    int h = blockIdx.x;

    // get the query vector for this head
    const float* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        const float* k = key_cache + loff + t * dim + h * head_size;
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
        for (int t = 0; t < seq_len; t++)
            val += att[t] * value_cache[loff + t * dim + h * head_size + i];
        output[h * head_size + i] = val;
    }
}
// void gpu_MultiHeadAttention(int pos, Config* p, RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
    // MultiHeadAttention_kernel <<<p->n_heads, num_threads_lrg>>> (pos, p->seq_len, s->q, s->att, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff, head_size * p->n_heads);
// }

void gpu_MultiHeadAttention(float *output, float *q, float *key_cache, float *value_cache, int num_heads, int head_size, int loff, int seq_len) {
    int dim = head_size * num_heads;
    MultiHeadAttention_kernel <<<num_heads, num_threads_lrg>>> (output, q, key_cache, value_cache, num_heads, head_size, loff, seq_len, dim);
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

__global__ void swiglu_kernel(float *shb, float *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = shb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= shb2[i];
        shb[i] = val;
    }
}
void gpu_swiglu(float *shb, float *shb2, int hidden_dim) {
    swiglu_kernel<<<divUp(hidden_dim, num_threads_med), num_threads_med>>>(shb, shb2, hidden_dim);
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

__global__ void accum_kernel(float* a, float* b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
      a[i] += b[i];
  }
}
void gpu_accum(float *a, float *b, int size) {
  accum_kernel<<<divUp(size, num_threads_med), num_threads_med>>>(a,b,size);
}

void accum(float *a, float *b, int size) {
  for (int i = 0; i < size; i++) {
    a[i] += b[i];
  }
}

// ----------------------------------------------------------

float* forward(Transformer* transformer, int token, int pos) {

  // a few convenience variables
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights;
  RunState* s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  float* content_row = w->token_embedding_table + token * dim;
#ifdef USE_GPU
  CHECK_HIP(hipMemcpy(x, content_row, dim*sizeof(*x), hipMemcpyHostToDevice));
#else
  memcpy(x, content_row, dim*sizeof(*x));
#endif

  // forward all the layers
  for(unsigned long long l = 0; l < p->n_layers; l++) {
    // printf("Layer: %llu\n", l);
    // attention rmsnorm
#ifdef USE_GPU
    gpu_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    gpu_matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
    gpu_matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
    gpu_matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

    gpu_RoPE(s->q, s->k, pos, dim, head_size, kv_dim);

    // save key,value at this time step (pos) to our kv cache
    float* key_cache_row = s->key_cache + loff + pos * dim;
    float* value_cache_row = s->value_cache + loff + pos * dim;
    CHECK_HIP(hipMemcpyAsync(key_cache_row, s->k, dim * sizeof(float), hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpyAsync(value_cache_row, s->v, dim * sizeof(float), hipMemcpyDeviceToDevice));
    gpu_MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos+1);


    // final matmul to get the output of the attention
    gpu_matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

    // residual connection back into x
    gpu_accum(x, s->xb2, dim);

    // ffn rmsnorm
    gpu_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    gpu_matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
    gpu_matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    gpu_swiglu(s->hb, s->hb2, hidden_dim);

    // final matmul to get the output of the ffn
    gpu_matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

    // residual connection
    gpu_accum(x, s->xb, dim);
#else
    rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // qkv matmuls for this position
    matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
    matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
    matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

    RoPE(s->q, s->k, pos, dim, head_size, kv_dim);

    MultiHeadAttention(pos, p, s, kv_dim, kv_mul, head_size, loff);

    // final matmul to get the output of the attention
    matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

    // residual connection back into x
    accum(x, s->xb2, dim);

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    swiglu(s->hb, s->hb2, hidden_dim);

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

    // residual connection
    accum(x, s->xb, dim);
#endif 
  }

  // final rmsnorm
#ifdef USE_GPU
  gpu_rmsnorm(x, x, w->rms_final_weight, dim);
  // classifier into logits
  gpu_matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
  CHECK_HIP(hipMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
#else
  rmsnorm(x, x, w->rms_final_weight, dim);
  matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
#endif 
  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens



int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
    if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer* t) {
  for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') { piece++; }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char*)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

void append_str(char *piece, std::string& str) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  //printf("%s", piece);
  str += piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = { .str = str }; // acts as the key to search for
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos) tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i=0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i < (*n_tokens-1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
      tokens[i] = tokens[i+1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling



int sample_argmax(float* probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
  ProbIndex* a_ = (ProbIndex*) a;
  ProbIndex* b_ = (ProbIndex*) b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
  free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

int sample_greedy(Sampler* sampler, float* logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  // greedy argmax sampling: take the token with the highest probability
  next = sample_argmax(logits, sampler->vocab_size);
  return next;
}

int sample_determin(const Sampler* sampler, float* logits, unsigned long long* rng_states, int idx) {
  // sample the token given the logits and some hyperparameters
  int next;
  float temperature = 1.0f;
  // apply the temperature to the logits
  for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= temperature; }
  // apply softmax to the logits to get the probabilities for next token
  softmax(logits, sampler->vocab_size);
  // flip a (float) coin (this is our source of entropy for sampling)
  float coin = random_f32(&rng_states[idx]);

  next = sample_mult(logits, sampler->vocab_size, coin);
  return next;
}


void build_requests(Requests* reqs, int num_reqs, int max_token_len, int max_seq_len) {
  reqs->num_reqs = num_reqs;
  reqs->max_token_len = max_token_len;
  reqs->max_seq_len = max_seq_len;
  reqs->str_reqs = (char*)calloc(num_reqs * max_token_len * max_seq_len + 1, sizeof(char));
  reqs->str_gens = (char*)calloc(num_reqs * max_token_len * max_seq_len + 1, sizeof(char));
  printf("requests size = %lu B\n", ((num_reqs * max_token_len * max_seq_len * sizeof(char) +1) * 2));
}

void free_requests(Requests* reqs) {
  free(reqs->str_reqs);
  free(reqs->str_gens);
}

char* get_str_req_ptr(Requests* reqs, int idx) {
  return reqs->str_reqs + idx * reqs->max_token_len * reqs->max_seq_len;
}

char* get_str_gen_ptr(Requests* reqs, int idx) {
  return reqs->str_gens + idx * reqs->max_token_len * reqs->max_seq_len;
}


int read_inputfile(const char* input_filename, int max_token_len, int max_seq_len, Requests* reqs) {
  std::string filename = input_filename;
  int num_reqs= 0;

  printf("max_token_len: %d, max_seq_len: %d\n", max_token_len, max_seq_len);

  std::ifstream openFile(filename.c_str());
  if (openFile.is_open() ) {
    std::string line;

    // Read the number of Requests
    std::getline(openFile, line);
    num_reqs = atoi(line.c_str());

    build_requests(reqs, num_reqs, max_token_len, max_seq_len);

    int idx = 0;
    while(std::getline(openFile, line)) {
      memcpy(get_str_req_ptr(reqs, idx), line.c_str(), line.size());
      idx++;
      if(idx >= num_reqs) break;
    }
    openFile.close();
  }
  else {
    fprintf(stderr, "cannot open the file: %s\n", input_filename);
    exit(EXIT_FAILURE);
  }

  return 0;
}

int write_outputfile(const char* output_filename, Requests* reqs) {
  std::string filename = output_filename;

  // write File
  std::ofstream writeFile(filename.c_str());
  if( writeFile.is_open() ){
    writeFile << reqs->num_reqs << "\n";
    for(int i = 0; i < reqs->num_reqs; i++) {
      writeFile << get_str_gen_ptr(reqs, i) << "\n";
    }
    writeFile.close();
  }
  else {
    fprintf(stderr, "cannot write the file: %s\n", output_filename);
    exit(EXIT_FAILURE);
  }

  return 0;
}



// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = (char*)"";
  if (prompt == NULL) { prompt = empty_prompt; }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;  // used to time our code, only initialized after first iteration
  int next;        // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;     // position in the sequence
  while (pos < steps) {
    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) { 
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    char* piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) { start = time_in_ms(); }

  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
    char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;        // will store the next token in the sequence
  int token;       // stores the current token to feed into the transformer
  int prev_token;
  int pos = 0;     // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) { user_turn = 1; }

    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      // the Assistant is responding, so print its output
      char* piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//
int test(Transformer *transformer, Tokenizer *tokenizer, Requests * requests, int batch=1, int start_idx=0, int end_idx=0) {
  // Count the number of the generated tokens
  int gen_cnt = 0;

  // Avoid randomness to generate tokens for batch input
  // Each input request has its Sampler each
  Sampler samplers[requests->num_reqs];
  for(int idx = start_idx; idx < end_idx; idx++) {
    build_sampler(&samplers[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);
  }

  // Loop for the multiple requests
  for(int idx = start_idx; idx < end_idx; idx++) {
    std::string gen_str = "";
    char* prompt = get_str_req_ptr(requests, idx);
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
      fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int steps = requests->max_seq_len; // max sequence length
    while (pos < steps) {
      // forward the transformer to get logits for the next token
      // printf("\npos: %d, token: %d\n", pos, token);
      float* logits = forward(transformer, token, pos);
      // printf("Pass forward\n");
      // advance the state machine
      if (pos < num_prompt_tokens - 1) {
        // if we are still processing the input prompt, force the next prompt token
        next = prompt_tokens[pos + 1];
      } else {
        // otherwise sample the next token from the logits
        next = sample(&samplers[idx], logits);
        //next = sample_greedy(sampler, logits);
        //next = sample_determin(sampler, logits, rng_states, idx);
      }
      pos++;

      // data-dependent terminating condition: the BOS (=1) token delimits sequences
      if (next == 1) { 
        break;
      }

      // print the token as string, decode it with the Tokenizer object
      char* piece = decode(tokenizer, token, next);
      // You don't need to print every tokens are generated.
      // {
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
      // }
      // gen_str += piece;
      append_str(piece, gen_str);
      token = next;
      // init the timer here because the first iteration can be slower
      // this timer is not important
      if (start == 0) { start = time_in_ms(); }

    }
    printf("\n");

    gen_str += "\n";
    strcpy(get_str_gen_ptr(requests, idx), gen_str.c_str());
    free(prompt_tokens);

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
      long end = time_in_ms();
      fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
      gen_cnt += pos-1;
    }
    printf("End of the request\n");
  }

  for(int idx = start_idx; idx < end_idx; idx++) {
    free_sampler(&samplers[idx]);
  }
  return gen_cnt;
}


int multi_test(
  Requests* requests,
  char* checkpoint_path=NULL,
  char* tokenizer_path = (char*)"tokenizer.bin",
  float temperature=1.0f,
  float topp=0.9f,
  int steps=256,
  unsigned long long rng_seed=0,
  char* input_filename=NULL,
  char* output_filename=NULL,
  int batch=1) { 

  int numDevices;
  CHECK_HIP(hipGetDeviceCount(&numDevices));

  int *num_gen_tokens = (int*)malloc(numDevices * sizeof(int));
#pragma omp parallel for num_threads(numDevices) schedule(static) shared(num_gen_tokens, requests)
  for (int d=0; d<numDevices; d++) {
    int device = omp_get_thread_num();
    CHECK_HIP(hipSetDevice(device));

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    
    // Split the requests into the number of devices
    printf("Build transformers from GPU %d\n", device);

    steps = transformer.config.seq_len;

    if(EXIT_FAILURE == read_inputfile(input_filename, tokenizer.max_token_length, steps, requests)) {
      fprintf(stderr, "cannot read input file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    int total_reqs = requests->num_reqs;

    int start = total_reqs / numDevices * device;
    int end = total_reqs / numDevices * (device + 1);
    if (device == numDevices - 1) end = total_reqs;
    num_gen_tokens[device] = test(&transformer, &tokenizer, requests, batch, start, end);
  }

  int gen_cnt = 0;
  for (int i = 0; i<numDevices; i++) {
    gen_cnt += num_gen_tokens[i];
  }

  return gen_cnt;

}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
// #ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Example: run model.bin -m test -f <input_filename> -o <output_filename>\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0 (ignore the arg for test mode)\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9 (ignore the arg for test mode)\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL) (ignore the arg for test mode)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len (for test mode steps = max_seq_len)\n");
  fprintf(stderr, "  -i <string> input prompt (ignore the arg for test mode)\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat|test, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -f <string> (only for test mode) input filename\n");
  fprintf(stderr, "  -o <string> (only for test mode) output filename\n");
  fprintf(stderr, "  -b <string> batch size\n");

  exit(EXIT_FAILURE);
}


#ifndef KERNEL_TEST

int main(int argc, char *argv[]) {
  printf("Enter main\n");
  // default parameters
  char *checkpoint_path = NULL;  // e.g. out/model.bin
  char *tokenizer_path = (char*)"tokenizer.bin";
  float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;            // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = (char*)"generate";    // generate|chat|test
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
  char *input_filename = NULL; // Input Filename
  char *output_filename = NULL; // Output Filename
  int batch = 1;

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
  for (int i = 2; i < argc; i+=2) {
    // do some basic validation
    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
    else if (argv[i][1] == 'f') { input_filename = argv[i + 1]; }
    else if (argv[i][1] == 'o') { output_filename = argv[i + 1]; }
    else if (argv[i][1] == 'b') { batch = atoi(argv[i + 1]); }
    else { error_usage(); }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  // Transformer transformer;
  // build_transformer(&transformer, checkpoint_path);
  // if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

  // // build the Tokenizer via the tokenizer .bin file
  // Tokenizer tokenizer;
  // build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // // build the Sampler
  // Sampler sampler;
  // build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  Requests requests;

  // run!
  if (strcmp(mode, "generate") == 0) {
    // generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } 
  else if (strcmp(mode, "chat") == 0) {
    //chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } 
  else if  (strcmp(mode, "test") == 0) {
    // int num_reqs;
    // steps = transformer.config.seq_len;
    if(input_filename == NULL || output_filename == NULL) {
      error_usage();
    }
    // if(EXIT_FAILURE == read_inputfile(input_filename, tokenizer.max_token_length, steps, &requests)) {
    //   fprintf(stderr, "cannot read input file: %s\n", input_filename);
    //   exit(EXIT_FAILURE);
    // }

    // Don't modify this parts for evaluation
    // {
    long start, end;
    start = time_in_ms();
    // int num_gen_tokens = test(&transformer, &tokenizer, &requests, batch, 0, requests.num_reqs);
    // printf("Prepare to run\n");
    int num_gen_tokens = multi_test(&requests,
      checkpoint_path, tokenizer_path, temperature, topp, steps, rng_seed, input_filename, output_filename, 
      batch);
    end = time_in_ms();

    // Your goal is to achieve best throughput(=reduce elapsed time)! 
    fprintf(stdout, "elapsed time(s): %f, achieved throughput(tok/s): %f\n", (double)(end-start)/1000, (num_gen_tokens) / (double)(end-start)*1000);
    //}

    if(EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
      fprintf(stderr, "cannot write output file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    free_requests(&requests);

  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  // free_sampler(&sampler);
  // free_tokenizer(&tokenizer);
  // free_transformer(&transformer);
  return 0;
}
#endif