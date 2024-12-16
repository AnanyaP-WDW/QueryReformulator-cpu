import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Disable gradient computation globally
torch.set_grad_enabled(False)

app = FastAPI(title="Query Reformulation API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

class QueryInput(BaseModel):
    query: str
    num_reformulations: int = 1

class QueryReformulator:
    def __init__(self, model_id):
        num_threads = min(os.cpu_count() - 1, 10)  # Use max 10 threads
        torch.set_num_threads(num_threads)
        
        # Set default tensor type to float16 for half precision
        torch.set_default_dtype(torch.float16)
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_id, use_fast=True)
        
        # Load model and convert to half precision
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model = self.model.half()  # Convert to half precision
        self.model.eval()

    def reformulate_query(self, input_sequence: str, nsent: int) -> List[str]:
        input_ids = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids
        
        results = []
        with torch.inference_mode():
            for _ in range(nsent):
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
        return results

# Initialize model at startup
MODEL_ID = "prhegde/t5-query-reformulation-RL"
reformulator = QueryReformulator(MODEL_ID)

# Warmup endpoint that will be called during container build
@app.get("/warmup")
def warmup():
    reformulator.reformulate_query("warm up query", nsent=1)
    return {"status": "warmed up"}

@app.post("/reformulate")
async def reformulate_query(query_input: QueryInput):
    try:
        start_time = time.time()
        results = reformulator.reformulate_query(
            query_input.query, 
            query_input.num_reformulations
        )
        execution_time = time.time() - start_time
        
        return {
            "reformulations": results,
            "execution_time_seconds": execution_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
