## Thought process and decisions
### time spent ~ 14-15 hrs
### Performance Target
- Latency: 100ms per query
- Deployment: Consumer-grade CPU environments
all test done on M1 macbook pro

### Model Selection Process
1) Query reformulation - query paraphasing - models already exist  - 
three options:
A) models already finetuned for query reformulation
B) General purpose models which can be prompted to do query reformulation - gpt2 , bart, mamba (state space model)
C) General purpose tiny decoder only models which can be finetuned for query reformualtion - not enough time, data, compute to train a more specialised task specific model - therefore did not take this appraoch

Preliminary results for B) were not satisfactory - too much inference time, bigger model size
So after testing ~10 models, I chose prhegde/t5-query-reformulation-RL -850 mb (T5 base finetuned on query reformulation) - got the best possible results with a model size that can potentially be reduced to achieve the desired latency of 100 ms 

example output:
prhegde/t5-query-reformulation-RL -850 mb
    Input: Create a table for top noise cancelling headphones that are not expensive
    Target: can you get noise cancelling headphones for less than $200
    Execution time: 1.7937488555908203 seconds

### Attempted Approaches
1. ✅ PyTorch (baseline)
   - Inference time: 1.7-1.9s on T5-base - can be optimized further
   - Hardware: M1 MacBook Pro (16GB RAM)

2. ❌ FastT5
   - Status: Failed
   - Issue: Compatibility problems between ONNX runtime and SentencePiece - library not maintained

3. ❌ ONNX Export + Quantization
   - Status: Failed
   - Issues: Model parameter breaks during quantization

4. ❌ transformer-deploy
   - Status: Not implemented
   - Reason: Implementation complexity

5. ❌ llama-ccp-server
   - Status: Server works but build breaks for the model
   - Reason: Could not get a stable onnx model

6. ❌ GPT 2 - ONNX Export + Quantization
   - Status: Failed
   - Issues: Model outputs are un-satisfactory
   - Note: refer to src/q_gpt2.py

7. ❌ FAT5-small-flan-en
   - Status: Not viable
   - Issues: Model lacks Language Model head
   - Note: Despite having Triton and Flash Attention optimizations

- T5 is an encoder-decoder model and was never designed to be fully compatible with float16 but with bfloat16 and float32 data types - therefore the qunatized models are unstable - more insights in approach.md

- my quantized models available on - https://huggingface.co/AnanyaPathak/t5-query-reformulation-RL-GGUF 

## Current Implementation and Optimizations 
please note that the results (both the quality of output and infernce time) are still not good.

### PyTorch-based Solution
The final implementation uses PyTorch with specific optimizations to achieve significant latency reduction.

### Implemented Optimizations

#### 1. Generation Parameters Optimization
- Limited maximum sequence length (128/64 tokens)
  - Prevents unnecessary computation for shorter queries
- Beam search optimization
  - Reduced beam width to 1
  - Minimizes parallel search paths
  - Maintains reasonable output quality
- Sampling parameters
  - do_sample=True (set to False for faster inference)
  - top_k=None (uncomment for faster inference)
  - top_p=None (uncomment for faster inference)
  - Tradeoff between output diversity and speed
- Half precision
  - Reduces memory usage and computation time
  - Uses float16 instead of float32

#### 2. Model Loading and Initialization
- Implemented model warm-up during build time
- Direct memory loading from disk
- Pre-loaded weights for faster inference
- Reduced cold-start latency

#### 3. Caching Strategy (Proposed)
- Concept: Cache key-value pairs for autoregressive generation
  - T5's autoregressive nature means K,V pairs remain constant
  - Potential to eliminate redundant computations
- Status: Not implemented due to complexity
- Future consideration for further optimization

#### 4. Torch JIT Compilation (Attempted)
- Challenge: Complex model architecture
- Issues:
  - Dual encoder-decoder inputs
  - Complex generation process
  - Tracing limitations with T5 architecture
- Status: Not viable for T5

#### 5. Thread and Evaluation Optimizations
- Model set to evaluation mode
- Optimized CPU thread allocation
- Implemented warm-up query system
- Environment specific tuning for M1 MacBook

### Performance Results
```
Before Optimization:
- Latency: ~1.5 seconds
- Variable performance

After Optimization:
- Latency: 0.2-0.3 seconds on m1 pro
- Consistent performance
- 80% reduction in response time
```

### Hardware Configuration
- Test Environment: M1 MacBook Pro
- RAM: 16GB
- CPU Threads: Optimized for M1 architecture

## usage
### Query Reformulation API

A FastAPI-based service that reformulates search queries using T5 model for better search results.

## Quick Start

1. Clone the repository:

2. build using docker compose:

I am building on mac m1 with 8 cpu. Please change the docker compose when emulating a consumer cpu:

ideal compose 
```bash
deploy:
      resources:
        limits:
          cpus: '10'  # Limit to 10 CPU cores to simulate consumer grade CPU
          memory: '16G'  # Typical consumer grade RAM
    environment:
      - MKL_NUM_THREADS=10
      - OMP_NUM_THREADS=10
      - OPENBLAS_NUM_THREADS=10
      - VECLIB_MAXIMUM_THREADS=10
      - NUMEXPR_NUM_THREADS=10
      - TOKENIZERS_PARALLELISM=false
```
please note:
 PyTorch's threading system (OpenMP) works best when the number of threads matches the number of physical CPU cores


3. Access the application:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs


### reflection and further improvements:
- encode decoder models like t5 have older architecture and therfero a lot of the newer optimzations dont work
- There are no finetunes available for "fast" query reformulation - gguf (other quantizations)
- scope for building a small yet specialed finetuned models based on modern arch - using flash attention v2 etc (also a lot of new optimizations) - llama 3.3 arch + RL (reward based policy)
- scope for maintaining fast t5 lib without the dependency hell
- using models with an encoder block is an overkill - only decoder model arch like gpt, llama 
- using MS deep speed for intel based CPU
