import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

class QueryReformulator:
    def __init__(self, model_id):
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.eval()

    def reformulate_query(self, input_sequence, nsent):
        input_ids = self.tokenizer(input_sequence, return_tensors="pt").input_ids
        print(f'Input: {input_sequence}')
        results = []
        with torch.no_grad():
            for i in range(nsent):
                output = self.model.generate(input_ids, max_length=35, num_beams=1, do_sample=True, repetition_penalty=1.8)
                target_sequence = self.tokenizer.decode(output[0], skip_special_tokens=True)
                results.append(target_sequence)
                print(f'Target: {target_sequence}')
        return results

if __name__ == "__main__":
    MODEL_ID = "prhegde/t5-query-reformulation-RL"
    input_sequence = "Create a table for top noise cancelling headphones that are not expensive"

    reformulator = QueryReformulator(MODEL_ID)
    
    start_time = time.time()
    reformulator.reformulate_query(input_sequence, nsent=1)
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time} seconds")
