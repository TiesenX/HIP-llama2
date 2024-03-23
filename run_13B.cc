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
#include "build_pipeline.h"
#include "kernels_pipeline.h"

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

float* forward(Transformer* transformer, int* token, int pos, int batch_size, int thread_id) {

  // a few convenience variables
  Config* p = &transformer->config;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  TransformerWeights* w[NUM_GPU];
  RunState* s[NUM_GPU];
  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    CHECK_HIP(hipSetDevice(device_id));
    w[device_id] = &transformer->weights_gpu[device_id];
    s[device_id] = &transformer->state[device_id][thread_id];
  }

  CHECK_HIP(hipSetDevice(0));
  float *x[NUM_GPU];
  x[0] = s[0]->x;

  if (thread_id > 0) {
    CHECK_HIP(hipStreamWaitEvent(streams[0], events[thread_id-1][0], 0));
    CHECK_HIP(hipEventSynchronize(events[thread_id-1][0]));
  }
  // copy the token embedding into x
  for (int idx = 0; idx < batch_size; idx++) {
    float* content_row = w[0]->token_embedding_table + token[idx] * dim;
    CHECK_HIP(hipMemcpyAsync(x[0] + idx * dim, content_row, dim*sizeof(*(x[0])), hipMemcpyDeviceToDevice, streams[0]));
    CHECK_HIP(hipStreamSynchronize(streams[0]));
  }

  CHECK_HIP(hipDeviceSynchronize());

  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    CHECK_HIP(hipSetDevice(device_id));

    if (device_id > 0) {
      CHECK_HIP(hipStreamWaitEvent(streams[device_id], events[thread_id][device_id-1], 0));
    }
    if (thread_id > 0) {
      CHECK_HIP(hipStreamWaitEvent(streams[device_id], events[thread_id-1][device_id], 0));
      CHECK_HIP(hipEventSynchronize(events[thread_id-1][device_id]));
    }

    x[device_id] = s[device_id]->x;

    // forward all the layers
    int num_layers = layer_end[device_id] - layer_begin[device_id];
    for(int l = 0; l < num_layers; l++) {
      
      // attention rmsnorm
      gpu_rmsnorm(s[device_id]->xb, x[device_id], w[device_id]->rms_att_weight + l*dim, dim, batch_size, streams[device_id]);

      // save key,value at this time step (pos) to our kv cache
      int loff = (layer_begin[device_id] + l) * p->seq_len * kv_dim * batch_size;
      s[device_id]->k = s[device_id]->key_cache + loff + pos * kv_dim * batch_size;
      s[device_id]->v = s[device_id]->value_cache + loff + pos * kv_dim * batch_size;

      // qkv matmuls for this position
      gpu_matmul(s[device_id]->q, s[device_id]->xb, w[device_id]->wq + l*dim*dim, dim, dim, batch_size, streams[device_id]);
      gpu_matmul(s[device_id]->k, s[device_id]->xb, w[device_id]->wk + l*dim*kv_dim, dim, kv_dim, batch_size, streams[device_id]);
      gpu_matmul(s[device_id]->v, s[device_id]->xb, w[device_id]->wv + l*dim*kv_dim, dim, kv_dim, batch_size, streams[device_id]);

      gpu_RoPE(s[device_id]->q, s[device_id]->k, pos, dim, head_size, kv_dim, batch_size, streams[device_id]);
      
      for (int idx = 0; idx < batch_size; idx++) {
        gpu_MultiHeadAttention(s[device_id]->xb + idx * dim, 
                               s[device_id]->q + idx * dim, 
                               s[device_id]->key_cache, s[device_id]->value_cache, 
                               kv_dim, kv_mul, p->n_heads, head_size, loff, pos+1, idx, batch_size, streams[device_id]);
      }
      
      // final matmul to get the output of the attention
      gpu_matmul(s[device_id]->xb2, s[device_id]->xb, w[device_id]->wo + l*dim*dim, dim, dim, batch_size, streams[device_id]);

      // residual connection back into x
      gpu_accum(x[device_id], s[device_id]->xb2, dim, batch_size, streams[device_id]);

      // ffn rmsnorm
      gpu_rmsnorm(s[device_id]->xb, x[device_id], w[device_id]->rms_ffn_weight + l*dim, dim, batch_size, streams[device_id]);

      // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
      // first calculate self.w1(x) and self.w3(x)
      gpu_matmul(s[device_id]->hb, s[device_id]->xb, w[device_id]->w1 + l*dim*hidden_dim, dim, hidden_dim, batch_size, streams[device_id]);
      gpu_matmul(s[device_id]->hb2, s[device_id]->xb, w[device_id]->w3 + l*dim*hidden_dim, dim, hidden_dim, batch_size, streams[device_id]);

      // SwiGLU non-linearity
      gpu_swiglu(s[device_id]->hb, s[device_id]->hb2, hidden_dim, batch_size, streams[device_id]);

      // final matmul to get the output of the ffn
      gpu_matmul(s[device_id]->xb, s[device_id]->hb, w[device_id]->w2 + l*dim*hidden_dim, hidden_dim, dim, batch_size, streams[device_id]);

      // residual connection
      gpu_accum(x[device_id], s[device_id]->xb, dim, batch_size, streams[device_id]);
    }

    if (device_id < NUM_GPU - 1) {
      CHECK_HIP(hipMemcpyAsync(s[device_id + 1]->x, s[device_id]->x, 
                          dim * batch_size * sizeof(float), hipMemcpyDeviceToDevice, streams[device_id]));
      CHECK_HIP(hipStreamSynchronize(streams[device_id]));
      CHECK_HIP(hipEventRecord(events[thread_id][device_id], streams[device_id]));
    }
    else {
      // final rmsnorm
      gpu_rmsnorm(x[device_id], x[device_id], w[device_id]->rms_final_weight, dim, batch_size, streams[device_id]);

      // classifier into logits
      gpu_matmul(s[device_id]->logits_gpu, x[device_id], w[device_id]->wcls, p->dim, p->vocab_size, batch_size, streams[device_id]);
      
      CHECK_HIP(hipMemcpyAsync(s[device_id]->logits, s[device_id]->logits_gpu, p->vocab_size * batch_size * sizeof(float), hipMemcpyDeviceToHost, streams[device_id]));
      CHECK_HIP(hipStreamSynchronize(streams[device_id]));
      CHECK_HIP(hipEventRecord(events[thread_id][device_id], streams[device_id]));
      return s[device_id]->logits;
    }
  }
  return NULL;
}


// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int cnt_threads[MAX_THREAD];

void* test_worker(void* args) {
  // printf("adhashdjashdkjahsd");
  thread_args* targs = (thread_args*)args;
  Transformer* transformer = targs->transformer;
  Tokenizer* tokenizer = targs->tokenizer;
  Requests* requests = targs->requests;
  int thread_id = targs->thread_id;
  int *next_req = targs->next_req;

  int current_req;
  int gen_cnt = 0;
  // int total_reqs = BATCH_SIZE * 2;
  int total_reqs = targs->total_reqs;

  while(true) {
    pthread_mutex_lock(&mutex);
    current_req = *next_req;
    int current_batch_size = BATCH_SIZE;
    if (current_req + current_batch_size > total_reqs) {
        current_batch_size = total_reqs - current_req;
    }
    *next_req = *next_req + current_batch_size;
    pthread_mutex_unlock(&mutex);
    if (current_req >= total_reqs) {
      break;
    }

    // Avoid randomness to generate tokens for batch input
    // Each input request has its Sampler each
    Sampler sampler[current_batch_size];
    // Loop for the multiple requests
    std::string gen_str[current_batch_size];
    char* prompt[current_batch_size];
    int* prompt_tokens[current_batch_size];
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens[current_batch_size];
    int next[current_batch_size];
    int token[current_batch_size];
    bool end_request[current_batch_size];
    int cnt_tokens[current_batch_size];

    // #pragma omp parallel for
    for (int idx = 0; idx < current_batch_size; idx++) {
      end_request[idx] = false;
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
      // this part for the fist GPU: continuous receive input

      // this part for the middle GPUs: continuous forward
      // -------------------------------------------------------------
      float* logits = forward(transformer, token, pos, current_batch_size, thread_id);
      // printf("Pass forward\n");
      
      // #pragma omp parallel for
      for (int idx = 0; idx < current_batch_size; idx++) {
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
      
      // #pragma omp parallel for
      for (int idx = 0; idx < current_batch_size; idx++) {
        if ((next[idx] == 1 || next[idx] == 2) && !end_request[idx]) {
          end_request[idx] = true;
        }
      }

      bool stop = true;
      for (int idx = 0; idx < current_batch_size; idx++) {
        if (!end_request[idx]) {
          stop = false;
        }
      }
      if (stop) break;

      // print the token as string, decode it with the Tokenizer object
      char* piece[current_batch_size];
      
      // #pragma omp parallel for
      for (int idx = 0; idx < current_batch_size; idx++) {
        if (!end_request[idx]){
          piece[idx] = decode(tokenizer, token[idx], next[idx]);
          // You don't need to print every tokens are generated.
          // {
          safe_printf(piece[idx]); // same as printf("%s", piece), but skips "unsafe" bytes
          fflush(stdout);
          // }
          // gen_str += piece;
          append_str(piece[idx], gen_str[idx]);
          token[idx] = next[idx];
          cnt_tokens[idx]++;
        }
      }

      // init the timer here because the first iteration can be slower
      // this timer is not important
      if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // #pragma omp parallel for
    for (int idx = 0; idx < current_batch_size; idx++) {
      gen_str[idx] += "\n";
      strcpy(get_str_gen_ptr(requests, current_req + idx), gen_str[idx].c_str());
      free(prompt_tokens[idx]);
      free_sampler(&sampler[idx]);
    }

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
      long end = time_in_ms();
      int sum_tokens = 0;
      for (int idx = 0; idx < current_batch_size; idx++) {
        printf("\nRequest %d: %d tokens\n", current_req + idx, cnt_tokens[idx]);
        sum_tokens += cnt_tokens[idx];
      }
      fprintf(stderr, "achieved tok/s: %f\n", sum_tokens / (double)(end-start)*1000);
      // gen_cnt += (pos - 1) * current_batch_size;
      gen_cnt += sum_tokens;
    }
    printf("End of the request\n");
  }

  cnt_threads[thread_id] = gen_cnt;
  return NULL; 
}


int test(Transformer *transformer, Tokenizer *tokenizer, Requests * requests, int batch=1) {
  // Count the number of the generated tokens
  int gen_cnt = 0;
  int num_reqs = requests->num_reqs;
  int* next_req = (int*)malloc(sizeof(int));
  
  *next_req = 0; // global request index

  thread_args args[MAX_THREAD];
  pthread_t threads[MAX_THREAD];
  pthread_attr_t attr[MAX_THREAD];
  cpu_set_t cpus[MAX_THREAD];

  printf("num_reqs: %d\n", num_reqs);

  for (int j=0; j<MAX_THREAD; j++) {

    args[j].transformer = transformer;
    args[j].tokenizer = tokenizer;
    args[j].requests = requests;
    args[j].thread_id = j;
    args[j].total_reqs = num_reqs;
    args[j].next_req = next_req;

    pthread_attr_init(&attr[j]);
    CPU_ZERO(&cpus[j]);
    CPU_SET(j, &cpus[j]);

    pthread_attr_setaffinity_np(&attr[j], sizeof(cpu_set_t), &cpus[j]);

    int rc = pthread_create(&threads[j], &attr[j], test_worker, args + j);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  for (int j=0; j<MAX_THREAD; j++) {
    pthread_join(threads[j], NULL);
  }

  for (int j=0; j<MAX_THREAD; j++) {
    gen_cnt += cnt_threads[j];
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

    // free_requests(&requests);

  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // destroyStreams();
  // // memory and file handles cleanup
  // free_sampler(&sampler);
  // free_tokenizer(&tokenizer);
  // free_transformer(&transformer);
  return 0;
}
#endif