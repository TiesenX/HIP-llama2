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

void test_rmsnorm(float* o, float* x, float* weight, int size) {
    float *cpu_output = o;
    float *gpu_output;
    CHECK_HIP(hipHostMalloc((void**)&gpu_output, size * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(gpu_output, o, size * sizeof(float), hipMemcpyHostToDevice));

    rmsnorm(cpu_output, x, weight, size);
    gpu_rmsnorm(gpu_output, x, weight, size);
    CHECK_HIP(hipDeviceSynchronize());

    for(int i = 0; i < size; i++) {
        if (fabs(cpu_output[i] - gpu_output[i]) > 1e-3) {
            fprintf(stderr, "RMSNORM failed at index %d, cpu: %f, gpu: %f\n", i, cpu_output[i], gpu_output[i]);
            exit(EXIT_FAILURE);
        }
    }
}

void test_matmul(float* xout, float* x, float* w, int n, int d) {
    float *cpu_output = xout;
    float *gpu_output;
    CHECK_HIP(hipHostMalloc((void**)&gpu_output, d * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(gpu_output, xout, d * sizeof(float), hipMemcpyHostToDevice));

    matmul(cpu_output, x, w, n, d);
    gpu_matmul(gpu_output, x, w, n, d);
    CHECK_HIP(hipDeviceSynchronize());

    for(int i = 0; i < d; i++) {
        if (fabs(cpu_output[i] - gpu_output[i]) > 1e-3) {
            fprintf(stderr, "MATMUL failed at index %d, cpu: %f, gpu: %f\n", i, cpu_output[i], gpu_output[i]);
            exit(EXIT_FAILURE);
        }
    }
}

void test_RoPE(float* sq, float* sk, int pos, int dim, int head_size, int kv_dim) {
    float *cpu_sq = sq, *cpu_sk = sk;
    float *gpu_sq, *gpu_sk;

    CHECK_HIP(hipHostMalloc((void**)&gpu_sq, dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void**)&gpu_sk, kv_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(gpu_sq, sq, dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(gpu_sk, sk, kv_dim * sizeof(float), hipMemcpyHostToDevice));

    RoPE(cpu_sq, cpu_sq, pos, dim, head_size, kv_dim);
    gpu_RoPE(gpu_sq, gpu_sk, pos, dim, head_size, kv_dim);
    CHECK_HIP(hipDeviceSynchronize());

    for(int i = 0; i < dim; i++) {
        if (fabs(cpu_sq[i] - gpu_sq[i]) > 1e-3 || fabs(cpu_sk[i] - gpu_sk[i]) > 1e-3) {
            fprintf(stderr, "RoPE failed at index %d, cpu_sq: %f, gpu_sq: %f, cpu_sk: %f, gpu_sk: %f\n", i, cpu_sq[i], gpu_sq[i], cpu_sk[i], gpu_sk[i]);
            exit(EXIT_FAILURE);
        }
    }
}


void test_accum(float* a, float* b, int size) {
    float *cpu_output = a;
    float *gpu_output;
    CHECK_HIP(hipHostMalloc((void**)&gpu_output, size * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(gpu_output, a, size * sizeof(float), hipMemcpyHostToDevice));

    accum(cpu_output, b, size);
    gpu_accum(gpu_output, b, size);
    CHECK_HIP(hipDeviceSynchronize());

    for(int i = 0; i < size; i++) {
        if (fabs(cpu_output[i] - gpu_output[i]) > 1e-3) {
            fprintf(stderr, "accum failed at index %d, cpu: %f, gpu: %f\n", i, cpu_output[i], gpu_output[i]);
            exit(EXIT_FAILURE);
        }
    }
}

void test_swiglu(float* shb, float* shb2, int hidden_dim) {
    float *cpu_shb = shb;
    float *gpu_shb;
    CHECK_HIP(hipHostMalloc((void**)&gpu_shb, hidden_dim * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpy(gpu_shb, shb, hidden_dim * sizeof(float), hipMemcpyHostToDevice));

    swiglu(cpu_shb, shb2, hidden_dim);
    gpu_swiglu(gpu_shb, shb2, hidden_dim);
    CHECK_HIP(hipDeviceSynchronize());

    for(int i = 0; i < hidden_dim; i++) {
        if (fabs(cpu_shb[i] - gpu_shb[i]) > 1e-3) {
            fprintf(stderr, "swiglu failed at index %d, cpu_shb: %f, gpu_shb: %f\n", i, cpu_shb[i], gpu_shb[i]);
            exit(EXIT_FAILURE);
        }
    }
}

float* forward_test(Transformer* transformer, int token, int pos) {
    
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
   
    CHECK_HIP(hipMemcpy(x, content_row, dim*sizeof(*x), hipMemcpyHostToDevice));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // printf("Layer: %llu\n", l);
        // attention rmsnorm
        test_rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        // rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        
        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        test_matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        test_matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        test_matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        test_RoPE(s->q, s->k, pos, dim, head_size, kv_dim);

        MultiHeadAttention(pos, p, s, kv_dim, kv_mul, head_size, loff);

        test_matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        test_accum(x, s->xb2, dim);

        test_rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // first calculate self.w1(x) and self.w3(x)
        test_matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        test_matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        test_swiglu(s->hb, s->hb2, hidden_dim);

        // final test_matmul to get the output of the ffn
        test_matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        test_accum(x, s->xb, dim);
    }

    // final rmsnorm
    test_rmsnorm(x, x, w->rms_final_weight, dim);
    // classifier into logits
    test_matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    // CHECK_HIP(hipMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return s->logits;
}

//
int test_funcs(Transformer *transformer, Tokenizer *tokenizer, Requests * requests, int batch=1) {
  // Count the number of the generated tokens
  int gen_cnt = 0;

  // Avoid randomness to generate tokens for batch input
  // Each input request has its Sampler each
  Sampler samplers[requests->num_reqs];
  for(int idx = 0; idx < requests->num_reqs; idx++) {
    build_sampler(&samplers[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);
  }

  // Loop for the multiple requests
  for(int idx = 0; idx < requests->num_reqs; idx++) {
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
      float* logits = forward_test(transformer, token, pos);
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

  for(int idx = 0; idx < requests->num_reqs; idx++) {
    free_sampler(&samplers[idx]);
  }
  return gen_cnt;
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
    char *mode = (char*)"test";    // generate|chat|test
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
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    Requests requests;

    // run!
    int num_reqs;
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
    int num_gen_tokens = test_funcs(&transformer, &tokenizer, &requests, batch);
    end = time_in_ms();

    // Your goal is to achieve best throughput(=reduce elapsed time)! 
    fprintf(stdout, "elapsed time(s): %f, achieved throughput(tok/s): %f\n", (double)(end-start)/1000, (num_gen_tokens) / (double)(end-start)*1000);
    //}

    if(EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
        fprintf(stderr, "cannot write output file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    free_requests(&requests);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}