## Imports
from llama_cpp import Llama
import os

model_kwargs = {
  "n_ctx":512,    # Context length to use
  "n_threads":2,   # Number of CPU threads to use
  "n_gpu_layers":0,# Number of model layers to offload to GPU. Set to 0 if only using CPU
}

# print(os.getcwd())

# if os.path.exists("models/t5-query-reformulation-RL.gguf"):
#     print("yes")
# else:
#     print("no")


## Instantiate model from downloaded file
llm = Llama(model_path="models/t5-query-reformulation-RL-tq2_0.gguf", **model_kwargs, chat_format="llama-2")

## Generation kwargs
generation_kwargs = {
    "max_tokens":50, # Max number of new tokens to generate
    "stop":["<|endoftext|>", "</s>"], # Text sequences to stop generation on
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}

## Run inference
prompt = "reformulate the following text query: The meaning of life is?"
res = llm(prompt, **generation_kwargs) # Res is a dictionary

## Unpack and the generated text from the LLM response dictionary and print it
print(res["choices"][0]["text"])