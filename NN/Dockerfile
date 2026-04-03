FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install ML dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    datasets>=2.19.0 \
    accelerate>=0.30.0 \
    trl>=1.0.0 \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.34.0"

# Pre-download base model during build (cached in image)
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)"

# Copy project files
COPY config.py service.py inference.py ./

# Copy trained LoRA adapter
COPY output/final/ ./output/final/

EXPOSE 8001

CMD ["python", "service.py", "--host", "0.0.0.0", "--port", "8001"]
