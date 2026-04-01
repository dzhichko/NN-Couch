"""
Interactive chat with the fine-tuned coaching model.

Usage:
    python inference.py                    # Use fine-tuned model
    python inference.py --base             # Use base model (for comparison)
    python inference.py --adapter PATH     # Custom adapter path
"""

import argparse
import re
import torch
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
    REPETITION_PENALTY,
)


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_model(adapter_path: str | None = None):
    compute_dtype = get_compute_dtype(BNB_4BIT_COMPUTE_DTYPE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    input_length = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = strip_think_blocks(response)

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with the coaching model")
    parser.add_argument(
        "--base", action="store_true",
        help="Use base model without LoRA adapter (for comparison)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help=f"Path to LoRA adapter (default: {OUTPUT_DIR}/final)",
    )
    args = parser.parse_args()

    if args.base:
        adapter_path = None
        print("Mode: BASE model (no fine-tuning)")
    else:
        adapter_path = args.adapter or f"{OUTPUT_DIR}/final"
        print(f"Mode: FINE-TUNED model (adapter: {adapter_path})")

    model, tokenizer = load_model(adapter_path)

    print("\n" + "=" * 60)
    print("  Online Coach - Interactive Chat")
    print("  Commands: /reset (new conversation), /quit (exit)")
    print("=" * 60 + "\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[Conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        response = generate_response(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": response})

        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    main()
