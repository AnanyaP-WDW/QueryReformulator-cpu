import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import os

# Disable gradient computation globally
torch.set_grad_enabled(False)

class QueryReformulator:
    def __init__(self, model_id):
        # Set number of CPU threads
        num_threads = os.cpu_count() - 1  # Leave one core free for system
        torch.set_num_threads(num_threads)
        
        # Set default tensor type to float16
        torch.set_default_dtype(torch.float16)
        
        # Initialize tokenizer with fast version
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, use_fast=True)
        
        # Load and convert model to half precision
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model = self.model.half()
        self.model.eval()  # Set to evaluation mode

    def reformulate_query(self, input_sequence, nsent):
        # Pre-allocate input tensor
        input_ids = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Set appropriate max length
        ).input_ids
        
        print(f'Input: {input_sequence}')
        results = []
        
        # Use inference_mode instead of no_grad (slightly faster)
        with torch.inference_mode():
            for i in range(nsent):
                output = self.model.generate(
                    input_ids,
                    max_length=35,
                    num_beams=1,
                    do_sample=False,  # Disable sampling for faster inference
                    top_k=None,       # Disable top_k sampling
                    top_p=None,       # Disable top_p sampling
                    repetition_penalty=1.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                target_sequence = self.tokenizer.decode(output[0], skip_special_tokens=True)
                results.append(target_sequence)
                print(f'Target: {target_sequence}')
        return results

if __name__ == "__main__":
    MODEL_ID = "prhegde/t5-query-reformulation-RL"
    input_sequence = "Create a table for top noise cancelling headphones that are not expensive"

    reformulator = QueryReformulator(MODEL_ID)
    
    # Warm-up run to ensure JIT compilation is complete
    reformulator.reformulate_query("warm up query", nsent=1)
    
    # Benchmark actual query
    start_time = time.time()
    reformulator.reformulate_query(input_sequence, nsent=1)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time} seconds")
