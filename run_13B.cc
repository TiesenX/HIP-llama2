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

float* forward(Transformer* transformer, int token, int pos) {

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
    s[device_id] = &transformer->state[device_id];
  }

  CHECK_HIP(hipSetDevice(0));
  float *x[NUM_GPU];
  x[0] = s[0]->x;

  // copy the token embedding into x
  float* content_row = w[0]->token_embedding_table + token * dim;

  CHECK_HIP(hipMemcpyAsync(x[0], content_row, dim*sizeof(*x[0]), hipMemcpyDeviceToDevice, streams[0]));
  CHECK_HIP(hipStreamSynchronize(streams[0]));

  for (int device_id = 0; device_id < NUM_GPU; device_id++) {
    CHECK_HIP(hipSetDevice(device_id));
    if (device_id > 0) {
      CHECK_HIP(hipStreamWaitEvent(streams[device_id], events[device_id - 1], 0));
    }

    x[device_id] = s[device_id]->x;

    // forward all the layers
    int num_layers = layer_end[device_id] - layer_begin[device_id];
    for(int l = 0; l < num_layers; l++) {
      
      // attention rmsnorm
      gpu_rmsnorm(s[device_id]->xb, x[device_id], w[device_id]->rms_att_weight + l*dim, dim, streams[device_id]);

      // save key,value at this time step (pos) to our kv cache
      int loff = (layer_begin[device_id] + l) * p->seq_len * kv_dim;
      s[device_id]->k = s[device_id]->key_cache + loff + pos * kv_dim;
      s[device_id]->v = s[device_id]->value_cache + loff + pos * kv_dim;

      // qkv matmuls for this position
      gpu_matmul(s[device_id]->q, s[device_id]->xb, w[device_id]->wq + l*dim*dim, dim, dim, streams[device_id]);
      gpu_matmul(s[device_id]->k, s[device_id]->xb, w[device_id]->wk + l*dim*kv_dim, dim, kv_dim, streams[device_id]);
      gpu_matmul(s[device_id]->v, s[device_id]->xb, w[device_id]->wv + l*dim*kv_dim, dim, kv_dim, streams[device_id]);

      gpu_RoPE(s[device_id]->q, s[device_id]->k, pos, dim, head_size, kv_dim, streams[device_id]);
      
      gpu_MultiHeadAttention(s[device_id]->xb, s[device_id]->q, 
                             s[device_id]->key_cache, s[device_id]->value_cache, 
                             kv_dim, kv_mul, p->n_heads, head_size, loff, pos+1, streams[device_id]);

      // final matmul to get the output of the attention
      gpu_matmul(s[device_id]->xb2, s[device_id]->xb, w[device_id]->wo + l*dim*dim, dim, dim, streams[device_id]);

      // residual connection back into x
      gpu_accum(x[device_id], s[device_id]->xb2, dim, streams[device_id]);

      // ffn rmsnorm
      gpu_rmsnorm(s[device_id]->xb, x[device_id], w[device_id]->rms_ffn_weight + l*dim, dim, streams[device_id]);

      // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
      // first calculate self.w1(x) and self.w3(x)
      gpu_matmul(s[device_id]->hb, s[device_id]->xb, w[device_id]->w1 + l*dim*hidden_dim, dim, hidden_dim, streams[device_id]);
      gpu_matmul(s[device_id]->hb2, s[device_id]->xb, w[device_id]->w3 + l*dim*hidden_dim, dim, hidden_dim, streams[device_id]);

      // SwiGLU non-linearity
      gpu_swiglu(s[device_id]->hb, s[device_id]->hb2, hidden_dim, streams[device_id]);

      // final matmul to get the output of the ffn
      gpu_matmul(s[device_id]->xb, s[device_id]->hb, w[device_id]->w2 + l*dim*hidden_dim, hidden_dim, dim, streams[device_id]);

      // residual connection
      gpu_accum(x[device_id], s[device_id]->xb, dim, streams[device_id]);
    }

    if (device_id < NUM_GPU - 1) {
      CHECK_HIP(hipMemcpyAsync(s[device_id + 1]->x, s[device_id]->x, 
                              dim * sizeof(float), hipMemcpyDeviceToDevice, streams[device_id]));
      CHECK_HIP(hipEventRecord(events[device_id], streams[device_id]));
    }
    else {
      // final rmsnorm
      gpu_rmsnorm(x[device_id], x[device_id], w[device_id]->rms_final_weight, dim, streams[device_id]);

      // classifier into logits
      gpu_matmul(s[device_id]->logits_gpu, x[device_id], w[device_id]->wcls, p->dim, p->vocab_size, streams[device_id]);
      
      CHECK_HIP(hipMemcpyAsync(s[device_id]->logits, s[device_id]->logits_gpu, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost, streams[device_id]));
      CHECK_HIP(hipStreamSynchronize(streams[device_id]));
      return s[device_id]->logits;
    }
  }
  return NULL;
}


// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//

int test(Transformer *transformer, Tokenizer *tokenizer, Requests * requests, int batch=1) {
  // Count the number of the generated tokens
  int gen_cnt = 0;

  Sampler samplers[requests->num_reqs];
  for(int idx = 0; idx < requests->num_reqs; idx++) {
    build_sampler(&samplers[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);
  }

  for(int idx = 0; idx < 1; idx++) {
  // for(int idx = 0; idx < requests->num_reqs; idx++) {
    // Loop for the multiple requests
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

      // -------------------------------------------------------------
      // Divide device and run in GPUs here
      float* logits = forward(transformer, token, pos);
      // float* logits = 0;
      // End of GPUs run
      // -------------------------------------------------------------

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
      if (next == 1 || next == 2) { 
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
  }

  for(int idx = 0; idx < requests->num_reqs; idx++) {
    free_sampler(&samplers[idx]);
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