import argparse

import uvicorn
import re
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY
)

# --- Global model state ---
model = None
tokenizer = None


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    return text.strip()


def load_model_and_tokenizer(adapter_path: str | None = None):
    compute_dtype = get_compute_dtype(BNB_4BIT_COMPUTE_DTYPE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    print(f"Loading base model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        mdl = PeftModel.from_pretrained(mdl, adapter_path)

    mdl.eval()
    print("Model ready!")
    return mdl, tok


def generate_response(messages: list[dict]) -> str:
    print("[NN] Step 1: getting device...")
    device = next(model.parameters()).device
    print(f"[NN] Step 2: device={device}, applying chat template...")
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    print(f"[NN] Step 3: input tokens={encoded['input_ids'].shape[1]}, starting model.generate()...")

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    print(f"[NN] Step 4: generation done, output tokens={output_ids.shape[1]}")
    input_length = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = strip_think_blocks(response)
    return response.strip()


# --- API schemas ---

class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]


class GenerateResponse(BaseModel):
    response: str


# --- FastAPI app ---

def create_app(adapter_path: str | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global model, tokenizer
        model, tokenizer = load_model_and_tokenizer(adapter_path)
        yield

    app = FastAPI(title="NN-Couch Model Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "adapter": adapter_path,
        }

    @app.get("/system-prompt")
    async def get_system_prompt():
        return {"system_prompt": SYSTEM_PROMPT}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")

        messages = [msg.model_dump() for msg in request.messages]
        print(f"[NN] Received generate request with {len(messages)} messages")
        import time
        start = time.time()
        response = generate_response(messages)
        elapsed = time.time() - start
        print(f"[NN] Generated response in {elapsed:.1f}s: {response[:80]}...")
        return GenerateResponse(response=response)

    return app


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NN-Couch Model Service")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--base", action="store_true", help="Use base model without adapter")
    parser.add_argument("--adapter", type=str, default=None)
    args = parser.parse_args()

    adapter = None if args.base else (args.adapter or f"{OUTPUT_DIR}/final")

    app = create_app(adapter)
    uvicorn.run(app, host=args.host, port=args.port)
