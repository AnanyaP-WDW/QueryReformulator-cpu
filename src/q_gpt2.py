import onnxruntime as ort
import numpy as np
import transformers
import os

class GPT2ONNXModel:
    def __init__(self, model_path):
        """
        Initialize the ONNX Runtime session with the quantized GPT-2 model
        
        Args:
            model_path (str): Path to the ONNX quantized model file
        """
        # Create an inference session
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )

    def generate_text(self, 
                      prompt, 
                      max_length=50, 
                      temperature=0.7, 
                      top_k=50):
        """
        Generate text using the quantized GPT-2 model
        
        Args:
            prompt (str): Input text to start generation
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
        
        Returns:
            str: Generated text
        """
        # Use GPT-2 tokenizer for preprocessing
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        
        # Encode the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors='np')
        
        # Create attention mask (1s for all tokens)
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        
        # Prepare generation parameters
        input_tensor = input_ids
        
        # Generation loop
        for _ in range(max_length):
            # Update attention mask for the current sequence length
            current_attention_mask = np.ones((1, input_tensor.shape[1]), dtype=np.int64)
            
            # Run inference with both inputs
            outputs = self.session.run(
                None, 
                {
                    "input_ids": input_tensor,
                    "attention_mask": current_attention_mask
                }
            )[0]
            
            # Get the last token predictions
            last_token_logits = outputs[0, -1, :]
            
            # Apply temperature scaling
            last_token_logits = last_token_logits / temperature
            
            # Get indices of top k values
            top_k_indices = np.argpartition(last_token_logits, -top_k)[-top_k:]
            # Get the corresponding logits
            top_k_logits = last_token_logits[top_k_indices]
            # Sort them in descending order
            sorted_indices = np.argsort(-top_k_logits)
            top_k_indices = top_k_indices[sorted_indices]
            top_k_logits = top_k_logits[sorted_indices]
            
            # Before exponentiating, subtract the maximum value for numerical stability
            top_k_logits = top_k_logits - np.max(top_k_logits)
            # Calculate exp values
            exp_logits = np.exp(top_k_logits)
            # Calculate probabilities
            probabilities = exp_logits / np.sum(exp_logits)

            # Add a safety check to ensure probabilities are valid
            if np.isnan(probabilities).any() or not np.isclose(np.sum(probabilities), 1.0):
                # If we still get invalid probabilities, fall back to uniform distribution
                probabilities = np.ones_like(probabilities) / len(probabilities)
            
            # Sample the next token
            next_token_index = np.random.choice(len(probabilities), p=probabilities)
            next_token = top_k_indices[next_token_index]
            
            # Append the new token
            input_tensor = np.column_stack((input_tensor, [next_token]))
            
            # Stop if end of sequence token
            if next_token == tokenizer.eos_token_id:
                break
        
        # Decode the generated text
        generated_text = tokenizer.decode(input_tensor[0])
        return generated_text

def main():
    # Path to your quantized ONNX model
    #print(os.getcwd())
    model_path = os.path.abspath(os.path.join(os.getcwd(), 'model/decoder_model_fp16.onnx'))
    print("model_path:", model_path)
    #model_path = '/models/decoder_model_fp16.onnx'
    
    # Initialize the model
    gpt2_model = GPT2ONNXModel(model_path)
    
    # Generate text
    prompt = "please reformulate the following query: Create a table for top noise cancelling headphones that are not expensive"
    generated_text = gpt2_model.generate_text(prompt)
    print("Generated Text:", generated_text)

if __name__ == '__main__':
    main()