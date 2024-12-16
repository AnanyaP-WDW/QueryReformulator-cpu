
## Napkin research notes

query reformulation -> essentially it's a paraphrasing task - need for an autoregressive decoder style model - size needs to me small -> t5 small/base etc.

### query reformulation models:
1) https://huggingface.co/prhegde/t5-query-reformulation-RL - https://github.com/PraveenSH/RL-Query-Reformulation
2) https://huggingface.co/harshasurampudi/t5-small-finetuned-query-reformulation
3) https://huggingface.co/prithivida/parrot_paraphraser_on_T5 - parrot paraphraser


## main issue 
T5 was never designed to be fully compatible with float16 but with bfloat16 and float32 data types


100ms latency on a consumer grade CPU -> onnnyx cpu optimization -> https://medium.com/@deeplch/the-beginners-guide-cpu-inference-optimization-with-onnx-99-8-tf-20-5-pytorch-speedup-83fd5cd38615

1) convert torch model to onnx interoperable model -> optimization (layer norm, node fusion? etc) -> t5 encoder destablized -> getting nan -> maybe beacuse older model operations werent menat to work with 4/8/16 data types -> more info -> https://medium.com/@kavika.roy/optimizing-the-t5-model-for-fast-inference-4a985cb597d1
2) quantization -> static or dynamic , quantization ? - 4 bit , 8 bit -> gguf -> https://huggingface.co/AnanyaPathak/t5-query-reformulation-RL-GGUF -> cannot troubleshoot 
3) Key Optimization Strategies:
    ONNX conversion reduces overhead
    Dynamic/static quantization reduces model size
    ONNX Runtime provides efficient CPU inference
    Use smaller model variants (base instead of large)
    Batch processing for better throughput - not relevant for one off cases
4) Latency Reduction ideas:
    Limit max_length in generation - 128/64? query dependent
    Use beam search with small beam width - beam = 1 -> less passes
    Warm up the model before production use - during build time loading model directly onto memory from disk

5) caching key value pairs associated with each token. since T5 is autoregressive and K,v remain same for the previous token, instead of recomputing can it be cached

6) Torch JIT ->  error occurs because the T5 model's forward pass requires both encoder and decoder inputs, and the generation process is more complex than what can be captured in a simple trace.

7) torch model eval + specify cpu threads + warmup query -> mac m1 - reduction of latency from ~1.5 sec to 0.2 - 0.3 

consumer grade cpu  example: -> 
Intel Core i5-13400/13600K (13th gen)| AMD Ryzen 5 7600/7600X (Zen 4)
Typical Specifications:
6-8 cores
12-16 threads
16GB DDR4

how to replicate a conmusmer grade cpu using docker -> local machine dependent? - yes


## finetuning dataset
mamba + peft -> https://github.com/nusnlp/paraphrasing-squad

## experiments that were tried:

1) fast t5 -> not maintained -> breaks -> onnx runtime and sentepeiecne are incompatible
2) vanilla pytorch -> 1.2 - 1.9 sec infernce on t5 base finetuned -> can't reach 100 ms even therreotically
3) onnx export + quantize + onnx runtime -> gpt 2 (decoder only) -> quality bad
4) onnx export + quantize + onnx runtime -> quantization breaks model param -> can't troubleshoot
5) transformer-deploy (https://github.com/ELS-RD/transformer-deploy/blob/main/demo/generative-model/t5.ipynb) -  too hard to understand
