"""
Interactive chat with the fine-tuned coaching model.

Usage:
    python inference.py                    # Use fine-tuned model
    python inference.py --base             # Use base model (for comparison)
    python inference.py --adapter PATH     # Custom adapter path
    python inference.py --add_think        # Add empty <think> block before assistant response (for DeepSeek-R1)
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
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)

# Try to import SYSTEM_PROMPT, fallback to default if not defined
try:
    from config import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = (
        "Ты — опытный лайф-коуч и эксперт по личностному развитию. Твоя задача — помогать пользователям ставить осмысленные цели, "
        "разбивать их на конкретные шаги, отслеживать прогресс и поддерживать мотивацию.\n\n"
        "Твои принципы коучинга:\n"
        "- Задавай уточняющие вопросы, чтобы понять истинные цели пользователя\n"
        "- Помогай разбивать большие цели на конкретные, измеримые, достижимые, релевантные и ограниченные по времени (SMART) задачи\n"
        "- Поддерживай и отмечай прогресс, каким бы малым он ни был\n"
        "- Предлагай практические стратегии для преодоления препятствий\n"
        "- Поддерживай ответственность пользователя, проявляя при этом сочувствие\n"
        "- Предлагай конкретные следующие шаги в конце каждого ответа\n"
        "- Будь тёплым, поддерживающим и прямолинейным"
    )


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def strip_think_blocks(text: str) -> str:
    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed <think> block (model didn't close the tag)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Remove any remaining think tags
    text = re.sub(r"</think>", "", text)
    return text.strip()


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
    add_think: bool = False,
) -> str:
    if add_think:
        gen_messages = messages.copy()
        pass

    # Apply chat template
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

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
    parser.add_argument(
        "--add_think", action="store_true",
        help="Add empty <think> block before assistant response (for DeepSeek-R1 models)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override base model name from config (e.g., 'Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R')",
    )
    args = parser.parse_args()

    if args.base:
        adapter_path = None
        print("Mode: BASE model (no fine-tuning)")
    else:
        adapter_path = args.adapter or f"{OUTPUT_DIR}/final"
        print(f"Mode: FINE-TUNED model (adapter: {adapter_path})")

    # If model override is provided, use it; otherwise from config
    model_name = args.model if args.model else MODEL_NAME



    model, tokenizer = load_model(adapter_path)

    print("\n" + "=" * 60)
    print("  Online Coach - Interactive Chat")
    if args.add_think:
        print("  (Empty <think> block will be added automatically)")
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

        response = generate_response(model, tokenizer, messages, add_think=args.add_think)
        messages.append({"role": "assistant", "content": response})

        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    main()