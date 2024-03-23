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
#include <pthread.h>
#include <math.h>
#include "build_schedule_4B.h"
#include "kernels_schedule_4B.h"

#include <ctime> // Include the header file for time functions

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

static hipStream_t streams[MAX_GPU][BATCH_SIZE];

void init_streams() {
  for(int i = 0; i < NUM_GPU; i++) {
    CHECK_HIP(hipSetDevice(i));
    for(int j = 0; j < BATCH_SIZE; j++) {
      CHECK_HIP(hipStreamCreate(&streams[i][j]));
    }
  }
}

void forward(Transformer *transformer, int *token, int *pos, int batch_size, float *logits, int device_id, int *new_batch_slot) {
  CHECK_HIP(hipSetDevice(device_id));
  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights_gpu[device_id];
  RunState *s = &transformer->state[device_id];
  float *x = s->x;

  // config variable (not related to selective batch)
  int dim = p->dim;
  int n_heads = p->n_heads;
  int kv_dim = (dim * p->n_kv_heads) / n_heads;
  int kv_mul = n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / n_heads;
  int vocab_size = p->vocab_size;
  int seq_len = p->seq_len;
  int n_layers = p->n_layers;

  // copy the token embedding into x
  for (int idx = 0; idx < batch_size; idx++) {
    float *content_row = w->token_embedding_table + token[idx] * dim;
    CHECK_HIP(hipMemcpyAsync(&x[idx * dim], content_row, dim * sizeof(*x), hipMemcpyDeviceToDevice));
  }
  CHECK_HIP(hipDeviceSynchronize());

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // attention rmsnorm
    gpu_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim, batch_size);

    // key and value point to the kv cache
    int loff = l * seq_len * BATCH_SIZE * kv_dim; // kv cache layer offset for convenience

    // qkv matmuls for this position
    gpu_matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim, batch_size);

    for (int batch = 0; batch < batch_size; batch++) {
      gpu_mat_vec(s->key_cache + loff + pos[batch] * BATCH_SIZE * kv_dim + new_batch_slot[batch] * kv_dim, 
                  s->xb + batch * dim, w->wk + l * dim * kv_dim, dim, kv_dim);
      gpu_mat_vec(s->value_cache + loff + pos[batch] * BATCH_SIZE * kv_dim + new_batch_slot[batch] * kv_dim, 
                  s->xb + batch * dim, w->wv + l * dim * kv_dim, dim, kv_dim);
    }

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    gpu_RoPE(s->q, s->key_cache + loff, pos, dim, head_size, kv_dim, new_batch_slot, batch_size);

    // multihead attention. iterate over all heads
    #pragma unroll
    for (int idx = 0; idx < batch_size; idx++) {
      gpu_MultiHeadAttention(s->xb + idx * dim, 
                             s->q + idx * dim, 
                             s->key_cache, s->value_cache, 
                             kv_dim, kv_mul, p->n_heads, head_size, loff, pos[idx]+1, new_batch_slot[idx], BATCH_SIZE, &streams[device_id][idx]);
    }
    for (int idx = 0; idx < batch_size; idx++) {
      CHECK_HIP(hipStreamSynchronize(streams[device_id][idx]));
    }

    // final matmul to get the output of the attention
    gpu_matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim, batch_size);

    // residual connection back into x
    gpu_accum(x, s->xb2, dim, batch_size);

    // ffn rmsnorm
    gpu_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim, batch_size);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    gpu_matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, batch_size);
    gpu_matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim, batch_size);

    // SwiGLU non-linearity
    gpu_swiglu(s->hb, s->hb2, hidden_dim, batch_size);

    // final matmul to get the output of the ffn
    gpu_matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, batch_size);

    // residual connection
    gpu_accum(x, s->xb, dim, batch_size);
  }

  // final rmsnorm
  gpu_rmsnorm(x, x, w->rms_final_weight, dim, batch_size);
  // classifier into logits
  gpu_matmul(logits, x, w->wcls, dim, vocab_size, batch_size);
  // CHECK_HIP(hipMemcpy(s->logits, s->logits_gpu, batch_size * vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());
  // return s->logits;
}


// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int current_req = 0;
int cnt_thread[MAX_GPU];

int** prompt_tokens;
int* num_prompt_tokens;
char** prompt;

void init_prompt_state(Requests* requests, Tokenizer* tokenizer){
  int num_reqs = requests->num_reqs;
  prompt_tokens = (int **)malloc(num_reqs * sizeof(int *));
  num_prompt_tokens = (int *)malloc(num_reqs * sizeof(int));
  prompt = (char **)malloc(num_reqs * sizeof(char *));

  for (int idx = 0; idx < num_reqs; idx++) {
    prompt[idx] = get_str_req_ptr(requests, idx);
    prompt_tokens[idx] = (int *)malloc((strlen(prompt[idx]) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    encode(tokenizer, prompt[idx], 1, 0, prompt_tokens[idx], &num_prompt_tokens[idx]);

    if (num_prompt_tokens[idx] < 1) {
      fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
    }
  }
}

void* test_in_thread(void *arg) {
  thread_args *args = (thread_args *)arg;
  Transformer *transformer = args->transformer; 
  Tokenizer *tokenizer = args->tokenizer; 
  Requests *requests = args->requests; 
  Sampler*samplers = args->samplers; 
  
  int device_id = args->device_id;
  int num_reqs = requests->num_reqs;
  int gen_cnt = 0;            // count generated tokens
  CHECK_HIP(hipSetDevice(device_id));

  int new_batch_slot[BATCH_SIZE];

  int req_in_batch[BATCH_SIZE]; // Point to indexes of currently processing requests
  int *pos_in_batch;    // Point to associated position of currently processing requests
  CHECK_HIP(hipHostMalloc((void **)&pos_in_batch, BATCH_SIZE * sizeof(int), hipMemAllocationTypePinned));
  int token[BATCH_SIZE];
  std::string gen_str[BATCH_SIZE];
  int next[BATCH_SIZE];


  pthread_mutex_lock(&mutex);
  int current_batch_size = BATCH_SIZE; // current requests may not fill all the batch
  if (current_req + current_batch_size > num_reqs) {
    current_batch_size = num_reqs - current_req;
  }
  for (int idx = 0; idx < current_batch_size; idx++) {
    new_batch_slot[idx] = idx;
    req_in_batch[idx] = current_req;
    pos_in_batch[idx] = 0;
    token[idx] = prompt_tokens[current_req][0];
    gen_str[idx] = "";
    current_req++;
  }
  pthread_mutex_unlock(&mutex);

  float *logits = transformer->state[device_id].logits;
  int max_seq_len = requests->max_seq_len;

  bool out_of_length[BATCH_SIZE] = {false};

  printf("Cur batch: %d\n", current_batch_size);
  
  // start the main loop
  while(true) {
    forward(transformer, token, pos_in_batch, current_batch_size, logits, device_id, new_batch_slot);

    // find out next token of each request in batch
    // #pragma omp parallel for
    for (int idx = 0; idx < current_batch_size; idx++) { 
      if (req_in_batch[idx] >= 0) {
        int req_idx = req_in_batch[idx];
        int pos = pos_in_batch[idx];
        if (pos < num_prompt_tokens[req_idx] - 1) {
          next[idx] = prompt_tokens[req_idx][pos + 1];
        } else {
          next[idx] = sample(&samplers[req_idx], logits + idx * transformer->config.vocab_size);
        }

        pos_in_batch[idx]++; // pos++
      }
    }
    // pos++;

    // Check end of any request in batch
    for (int idx = 0; idx < current_batch_size; idx++) {
      // if got end condition but not empty
      if ((pos_in_batch[idx] >= max_seq_len || next[idx] == 1 || next[idx] == 2) && req_in_batch[idx] >= 0) {

        int req_idx = req_in_batch[idx];
        
        if (pos_in_batch[idx] >= max_seq_len) {
          out_of_length[idx] = true;
          if (next[idx] != 1 && next[idx] != 2) {
            char *piece = decode(tokenizer, token[idx], next[idx]);
            // You don't need to print every tokens are generated.
            // {
            // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            // fflush(stdout);
            // }
            // gen_str += piece;
            append_str(piece, gen_str[idx]);
            pos_in_batch[idx]++;
          }
        }
        gen_str[idx] += "\n";
        strcpy(get_str_gen_ptr(requests, req_idx), gen_str[idx].c_str());

        printf("DONE REQ: %d\n", req_idx);
        fflush(stdout);

        free_sampler(&samplers[req_idx]);
        
        if (pos_in_batch[idx] > 1) {
          gen_cnt += pos_in_batch[idx] - 1;
        }

        // Add another req to batch
        pthread_mutex_lock(&mutex);
        if (current_req < num_reqs) {
          req_in_batch[idx] = current_req;
          current_req++;
          pthread_mutex_unlock(&mutex);

          token[idx] = prompt_tokens[req_in_batch[idx]][0];
          gen_str[idx] = "";

          pos_in_batch[idx] = 0;
        } else {
          pthread_mutex_unlock(&mutex);
          req_in_batch[idx] = -1; // when there is no more request to add to batch
        }
      }
    }

    // decode next token
    for (int idx = 0; idx < current_batch_size; idx++) {
      if (out_of_length[idx] || next[idx] == 1 || next[idx] == 2 || req_in_batch[idx] < 0) {
        continue;
      } else {
        char *piece = decode(tokenizer, token[idx], next[idx]);
        // You don't need to print every tokens are generated.
        // {
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        // }

        // gen_str += piece;
        append_str(piece, gen_str[idx]);
        token[idx] = next[idx];
      }
    }

    int new_batch_size = current_batch_size;
    for (int idx = 0; idx < current_batch_size; idx++) {
      out_of_length[idx] = false;
      if (req_in_batch[idx] == -1) {
        new_batch_size--;
        for (int j = idx + 1; j < current_batch_size; j++) {
          if (req_in_batch[j] != -1) {
            new_batch_size++;
            req_in_batch[idx] = req_in_batch[j];
            pos_in_batch[idx] = pos_in_batch[j];
            token[idx] = token[j];
            gen_str[idx] = gen_str[j];
            new_batch_slot[idx] = new_batch_slot[j];

            req_in_batch[j] = -1;
            break;
          }
        }
      }
    }
    if (new_batch_size == 0) break;
    current_batch_size = new_batch_size;
  }

  cnt_thread[device_id] = gen_cnt;
  return NULL;
}

int test(Transformer *transformer, Tokenizer *tokenizer, Requests *requests, Sampler *samplers, int batch = 1) {
  int gen_cnt = 0;
  // Thread argument for each thread
  thread_args arg[NUM_GPU];
  pthread_t threads[NUM_GPU];
  pthread_attr_t attr[NUM_GPU];
  cpu_set_t cpus[NUM_GPU];

  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    CHECK_HIP(hipSetDevice(device_id));
    arg[device_id].device_id = device_id;
    arg[device_id].requests = requests;
    arg[device_id].transformer = transformer;
    arg[device_id].tokenizer = tokenizer;
    arg[device_id].samplers = samplers;

    pthread_attr_init(&attr[device_id]);
    CPU_ZERO(&cpus[device_id]);
    CPU_SET(device_id, &cpus[device_id]);

    // limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[device_id], sizeof(cpu_set_t), &cpus[device_id]);

    int rc = pthread_create(&threads[device_id], &attr[device_id], test_in_thread, &arg[device_id]);
  }

  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    int rc = pthread_join(threads[device_id], NULL);
  }

  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    gen_cnt += cnt_thread[device_id];
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

#include <sys/time.h>

double get_mtime() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
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
  CHECK_HIP(hipGetDeviceCount(&NUM_GPU));
  printf("Number of Devices: %d\n", NUM_GPU);

  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  Requests requests;

  // run!
  if (strcmp(mode, "generate") == 0) {
    // generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } 
  else if (strcmp(mode, "chat") == 0) {
    //chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } 
  else if  (strcmp(mode, "test") == 0) {
    steps = transformer.config.seq_len;
    if(input_filename == NULL || output_filename == NULL) {
      error_usage();
    }
    if(EXIT_FAILURE == read_inputfile(input_filename, tokenizer.max_token_length, steps, &requests)) {
      fprintf(stderr, "cannot read input file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    // double elapsed_time = 0.0;
    // int batch_size = 2;
    // int n = transformer.config.dim;
    // float *xGPU;
    // float *wGPU;
    // int d = transformer.config.vocab_size;
    // float *xoutGPU;
    // CHECK_HIP(hipMalloc(&xGPU, n * batch_size * sizeof(float)));
    // CHECK_HIP(hipMalloc(&wGPU, n * d * sizeof(float)));
    // CHECK_HIP(hipHostMalloc(&xoutGPU, d * batch_size * sizeof(float), hipMemAllocationTypePinned));

    // // ...

    // for (int i = 0; i <= 10; ++i) {
    //   double start_time = get_mtime();
    //   gpu_matmul(xoutGPU, xGPU, wGPU, n, d, batch_size);
    //   CHECK_HIP(hipDeviceSynchronize());
    //   if (i > 0) {
    //     elapsed_time += get_mtime() - start_time; // Calculate the elapsed time
    //   }
    // }
    // printf("%f sec, GFLOPS: %f \n", elapsed_time, 2 * n * d * batch_size / (elapsed_time / 10) / 1e9);
    // fflush(stdout);

    // float *xoutGPU2;
    // CHECK_HIP(hipHostMalloc(&xoutGPU2, d * batch_size * sizeof(float), hipMemAllocationTypePinned));

    // // ...

    // for (int i = 0; i <= 10; ++i) {
    //   double start_time = get_mtime();
    //   gpu_matmul2(xoutGPU2, xGPU, wGPU, n, d, batch_size);
    //   CHECK_HIP(hipDeviceSynchronize());
    //   if (i > 0) {
    //     elapsed_time += get_mtime() - start_time; // Calculate the elapsed time
    //   }
    // }
    // printf("%f sec, GFLOPS: %f \n", elapsed_time, 2 * n * d * batch_size / (elapsed_time / 10) / 1e9);
    // fflush(stdout);

    // for (int i =0; i < d*batch_size; i++){
    //   if (xoutGPU[i] - xoutGPU2[i] > 1e-3){
    //     printf("Error at %d\n", i);
    //     break;
    //   }
    // }
    // // ...

    init_prompt_state(&requests, &tokenizer);
    init_streams();
    Sampler samplers[requests.num_reqs];
    for (int idx = 0; idx < requests.num_reqs; idx++)
    {
      build_sampler(&samplers[idx], transformer.config.vocab_size, 1.0f, 0.9f, 314028);
    }

    // Don't modify this parts for evaluation
    // {
    long start, end;
    start = time_in_ms();
    int num_gen_tokens = test(&transformer, &tokenizer, &requests, samplers, BATCH_SIZE);
    end = time_in_ms();

    // Your goal is to achieve best throughput(=reduce elapsed time)! 
    fprintf(stdout, "elapsed time(s): %f, achieved throughput(tok/s): %f\n", (double)(end-start)/1000, (num_gen_tokens) / (double)(end-start)*1000);

    if(EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
      fprintf(stderr, "cannot write output file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    free_requests(&requests);

  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // // memory and file handles cleanup
  // free_sampler(&sampler);
  // free_tokenizer(&tokenizer);
  // free_transformer(&transformer);
  return 0;
}