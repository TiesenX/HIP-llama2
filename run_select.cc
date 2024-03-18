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
#include "build_select.h"
#include "kernels_select.h"

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
// Forward inference

float* forward(Transformer* transformer, int* token, int pos, int device_id, int thread_id, hipStream_t *stream) {

  // a few convenience variables
  Config* p = &transformer->config;
  TransformerWeights* w = &transformer->weights_gpu[device_id];
  RunState* s = &transformer->state[device_id][thread_id];
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  for (int idx = 0; idx < BATCH_SIZE; idx++) {
    float* content_row = w->token_embedding_table + token[idx] * dim;
    CHECK_HIP(hipMemcpyAsync(&x[idx * dim], content_row, dim*sizeof(*x), hipMemcpyDeviceToDevice, *stream));
  }
  CHECK_HIP(hipStreamSynchronize(*stream));

  // forward all the layers
  for(unsigned long long l = 0; l < p->n_layers; l++) {
    // printf("Layer: %llu\n", l);
    // attention rmsnorm
    gpu_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim, BATCH_SIZE, stream);

    // save key,value at this time step (pos) to our kv cache
    int loff = l * p->seq_len * kv_dim * BATCH_SIZE; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim * BATCH_SIZE;
    s->v = s->value_cache + loff + pos * kv_dim * BATCH_SIZE;

    // qkv matmuls for this position
    gpu_matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim, BATCH_SIZE, stream);
    gpu_matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim, BATCH_SIZE, stream);
    gpu_matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim, BATCH_SIZE, stream);

    gpu_RoPE(s->q, s->k, pos, dim, head_size, kv_dim, BATCH_SIZE, stream);

    for (int idx = 0; idx < BATCH_SIZE; idx++) {
      gpu_MultiHeadAttention(s->xb + idx * dim, 
                             s->q + idx * dim, 
                             s->key_cache, s->value_cache, 
                             kv_dim, kv_mul, p->n_heads, head_size, loff, pos+1, idx, BATCH_SIZE, stream);
    }

    // final matmul to get the output of the attention
    gpu_matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim, BATCH_SIZE, stream);

    // residual connection back into x
    gpu_accum(x, s->xb2, dim, BATCH_SIZE, stream);

    // ffn rmsnorm
    gpu_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim, BATCH_SIZE, stream);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    gpu_matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim, BATCH_SIZE, stream);
    gpu_matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim, BATCH_SIZE, stream);

    // SwiGLU non-linearity
    gpu_swiglu(s->hb, s->hb2, hidden_dim, BATCH_SIZE, stream);

    // final matmul to get the output of the ffn
    gpu_matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim, BATCH_SIZE, stream);

    // residual connection
    gpu_accum(x, s->xb, dim, BATCH_SIZE, stream);
  }

  // final rmsnorm
  gpu_rmsnorm(x, x, w->rms_final_weight, dim, BATCH_SIZE, stream);
  // classifier into logits
  gpu_matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size, BATCH_SIZE, stream);
  CHECK_HIP(hipMemcpyAsync(s->logits, s->logits_gpu, p->vocab_size * BATCH_SIZE * sizeof(float), hipMemcpyDeviceToHost, *stream));

  return s->logits;
}

static hipStream_t streams[MAX_GPU][MAX_REQ];

void initStreams() {
  for(int i = 0; i < NUM_GPU; i++) {
    CHECK_HIP(hipSetDevice(i));
    for(int j = 0; j < MAX_REQ; j++) {
      CHECK_HIP(hipStreamCreate(&streams[i][j]));
    }
  }
}

void destroyStreams() {
  for(int i = 0; i < NUM_GPU; i++) {
    CHECK_HIP(hipSetDevice(i));
    for(int j = 0; j < MAX_REQ; j++) {
      CHECK_HIP(hipStreamDestroy(streams[i][j]));
    }
  }
}


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int cnt_threads[MAX_GPU][MAX_REQ];

void* test_worker(void* args) {
  // printf("adhashdjashdkjahsd");
  thread_args* targs = (thread_args*)args;
  Transformer* transformer = targs->transformer;
  Tokenizer* tokenizer = targs->tokenizer;
  Requests* requests = targs->requests;
  int device_id = targs->device_id;
  int thread_id = targs->thread_id;
  int *next_req = targs->next_req;

  int current_req;
  int gen_cnt = 0;
  
  while(true) {
    // Need handle when num req can not divide by batch size
    // --------------------------------------------------------------
    pthread_mutex_lock(&mutex);
    current_req = *next_req;
    *next_req = *next_req + BATCH_SIZE;
    pthread_mutex_unlock(&mutex);
    if (current_req >= targs->total_reqs) {
      break;
    }
    // --------------------------------------------------------------

    // Avoid randomness to generate tokens for batch input
    // Each input request has its Sampler each
    Sampler sampler[BATCH_SIZE];
    // Loop for the multiple requests
    std::string gen_str[BATCH_SIZE];
    char* prompt[BATCH_SIZE];
    int* prompt_tokens[BATCH_SIZE];
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens[BATCH_SIZE];
    int next[BATCH_SIZE];
    int token[BATCH_SIZE];
    int end_request[BATCH_SIZE];
    int cnt_tokens[BATCH_SIZE];

    for (int idx = 0; idx < BATCH_SIZE; idx++) {
      end_request[idx] = 0;
      cnt_tokens[idx] = 0;
      prompt[idx] = get_str_req_ptr(requests, current_req + idx);
      prompt_tokens[idx] = (int*)malloc((strlen(prompt[idx])+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
      gen_str[idx] = "";
      num_prompt_tokens[idx] = 0;

      build_sampler(&sampler[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);

      encode(tokenizer, prompt[idx], 1, 0, prompt_tokens[idx], &num_prompt_tokens[idx]);
      if (num_prompt_tokens[idx] < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
      }
      token[idx] = prompt_tokens[idx][0];
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int pos = 0;     // position in the sequence
    
    int steps = requests->max_seq_len; // max sequence length
    while (pos < steps) {
      // forward the transformer to get logits for the next token
      // printf("\npos: %d, token: %d\n", pos, token);
      float* logits = forward(transformer, token, pos, device_id, thread_id, &streams[device_id][thread_id]);
      CHECK_HIP(hipStreamSynchronize(streams[device_id][thread_id]));
      // printf("Pass forward\n");

      for (int idx = 0; idx < BATCH_SIZE; idx++) {
        // advance the state machine
        if (pos < num_prompt_tokens[idx] - 1) {
          // if we are still processing the input prompt, force the next prompt token
          next[idx] = prompt_tokens[idx][pos + 1];
        } else {
          // otherwise sample the next[idx] token from the logits
          next[idx] = sample(&sampler[idx], logits + idx * transformer->config.vocab_size);
          //next = sample_greedy(sampler, logits);
          //next = sample_determin(sampler, logits, rng_states, idx);
        }
      }
      pos++;

      // data-dependent terminating condition: the BOS (=1) token delimits sequences
      for (int idx = 0; idx < BATCH_SIZE; idx++) {
        if (next[idx] == 1 || next[idx] == 2) {
          end_request[idx] = 1;
          if (cnt_tokens[idx] == 0) cnt_tokens[idx] = pos;
        }
      }

      bool end = true;
      for (int idx = 0; idx < BATCH_SIZE; idx++) {
        if (end_request[idx] != 1) {
          end = false;
        }
      }
      if (end) break;

      // print the token as string, decode it with the Tokenizer object
      char* piece[BATCH_SIZE];
      
      for (int idx = 0; idx < BATCH_SIZE; idx++) {
        if (end_request[idx] != 1){
          piece[idx] = decode(tokenizer, token[idx], next[idx]);
          // You don't need to print every tokens are generated.
          // {
          // safe_printf(piece[idx]); // same as printf("%s", piece), but skips "unsafe" bytes
          fflush(stdout);
          // }
          // gen_str += piece;
          append_str(piece[idx], gen_str[idx]);
          token[idx] = next[idx];
        }
      }

      // init the timer here because the first iteration can be slower
      // this timer is not important
      if (start == 0) { start = time_in_ms(); }

    }
    printf("\n");

    for (int idx = 0; idx < BATCH_SIZE; idx++) {
      gen_str[idx] += "\n";
      strcpy(get_str_gen_ptr(requests, current_req + idx), gen_str[idx].c_str());
      free(prompt_tokens[idx]);
      free_sampler(&sampler[idx]);
    }

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
      long end = time_in_ms();
      int sum_tokens = 0;
      for (int idx = 0; idx < BATCH_SIZE; idx++) {
        sum_tokens += cnt_tokens[idx] - 1;
      }
      fprintf(stderr, "achieved tok/s: %f\n", sum_tokens / (double)(end-start)*1000);
      gen_cnt += sum_tokens;
    }
    printf("End of the request\n");
  }
  
  cnt_threads[device_id][thread_id] = gen_cnt;
  return NULL; 
}

// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//


int test(Transformer *transformer, Tokenizer *tokenizer, Requests * requests, int batch=1) {
  // Count the number of the generated tokens
  int gen_cnt = 0;
  int num_reqs = requests->num_reqs;
  int* next_req = (int*)malloc(sizeof(int));
  
  *next_req = 0; // global request index

  thread_args args[MAX_GPU][MAX_REQ];
  pthread_t threads[MAX_GPU][MAX_REQ];
  pthread_attr_t attr[MAX_GPU][MAX_REQ];
  cpu_set_t cpus[MAX_GPU][MAX_REQ];

  printf("num_reqs: %d\n", num_reqs);

  for (int i=0; i<NUM_GPU; i++) {
    // CHECK_HIP(hipSetDevice(i));
    for (int j=0; j<MAX_REQ; j++) {

      args[i][j].transformer = transformer;
      args[i][j].tokenizer = tokenizer;
      args[i][j].requests = requests;
      args[i][j].thread_id = j;
      args[i][j].device_id = i;
      args[i][j].total_reqs = num_reqs;
      args[i][j].next_req = next_req;

      pthread_attr_init(&attr[i][j]);
      CPU_ZERO(&cpus[i][j]);
      CPU_SET(i * MAX_REQ + j, &cpus[i][j]);

      pthread_attr_setaffinity_np(&attr[i][j], sizeof(cpu_set_t), &cpus[i][j]);

      int rc = pthread_create(&threads[i][j], &attr[i][j], test_worker, args[i] + j);
      if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }
  }

  for (int i=0; i<NUM_GPU; i++) {
    for (int j=0; j<MAX_REQ; j++) {
      pthread_join(threads[i][j], NULL);
    }
  }

  for (int i=0; i<NUM_GPU; i++) {
    for (int j=0; j<MAX_REQ; j++) {
      gen_cnt += cnt_threads[i][j];
    }
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
  CHECK_HIP(hipGetDeviceCount(&NUM_GPU));
  printf("Number of Devices: %d\n", NUM_GPU);

  initStreams();
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

    // Don't modify this parts for evaluation
    // {
    long start, end;
    start = time_in_ms();
    int num_gen_tokens = test(&transformer, &tokenizer, &requests, batch);
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

  destroyStreams();
  // // memory and file handles cleanup
  // free_sampler(&sampler);
  // free_tokenizer(&tokenizer);
  // free_transformer(&transformer);
  return 0;
}
#endif