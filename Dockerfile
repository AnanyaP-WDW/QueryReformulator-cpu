# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .
COPY src/main.py .
COPY src/static/ /app/static/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for CPU optimization
ENV MKL_NUM_THREADS=6 \
    OMP_NUM_THREADS=6 \
    OPENBLAS_NUM_THREADS=6 \
    VECLIB_MAXIMUM_THREADS=6 \
    NUMEXPR_NUM_THREADS=6 \
    TOKENIZERS_PARALLELISM=false

# Download and cache the model during build
RUN python3 -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; model_id='prhegde/t5-query-reformulation-RL'; T5ForConditionalGeneration.from_pretrained(model_id); T5Tokenizer.from_pretrained(model_id)"

# Run warmup query during build
RUN python3 -c "import main; main.warmup()"

# Expose port
EXPOSE 8000

# Run FastAPI with multiple workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 