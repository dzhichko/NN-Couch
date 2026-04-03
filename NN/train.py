
import os
import torch
import json
import time
import inspect
import uuid
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import trl
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import (
    MODEL_NAME,
    OUTPUT_DIR,
    DATASET_PATH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    NUM_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LR_SCHEDULER,
    WARMUP_RATIO,
    MAX_SEQ_LENGTH,
    MAX_GRAD_NORM,
    LOGGING_STEPS,
    SAVE_STEPS,
    FP16,
    BF16,
    GRADIENT_CHECKPOINTING,
    OPTIM,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ASSISTANT_ONLY_LOSS = False


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": "c5eaaf",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = Path(__file__).resolve().parent / "debug-c5eaaf.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main():
    run_id = f"pre-fix-{uuid.uuid4().hex[:8]}"
    print("=" * 60)
    print("  DeepSeek-R1 Coaching Fine-Tuning (QLoRA)")
    print("=" * 60)

    debug_log(
        run_id,
        "H_ENV_VERSIONS",
        "train.py:main:start",
        "runtime versions",
        {
            "python_torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "trl_version": trl.__version__,
        },
    )

    compute_dtype = get_compute_dtype(BNB_4BIT_COMPUTE_DTYPE)

    # --- Tokenizer ---
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- 4-bit quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # --- Model ---
    print(f"Loading model: {MODEL_NAME} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # --- LoRA ---
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Dataset ---
    print(f"\nLoading dataset: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'messages' column with conversational samples.")

    # Do not inject <think> blocks. Let the model learn the chat format directly.
    print(f"Dataset size: {len(dataset)} examples")

    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        report_to="none",
        save_total_limit=3,
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        assistant_only_loss=ASSISTANT_ONLY_LOSS,
        dataset_num_proc=1,
    )

    # --- Trainer ---
    print("\nInitializing SFTTrainer...")
    sft_signature = str(inspect.signature(SFTTrainer.__init__))
    debug_log(
        run_id,
        "H_SFT_SIGNATURE",
        "train.py:before_trainer_init",
        "SFTTrainer init signature",
        {"signature": sft_signature},
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "args": training_args,
    }

    debug_log(
        run_id,
        "H_UNEXPECTED_KWARG",
        "train.py:before_trainer_init",
        "SFTTrainer kwargs prepared",
        {
            "kwargs_keys": list(trainer_kwargs.keys()),
            "max_seq_length_type": type(MAX_SEQ_LENGTH).__name__,
            "max_seq_length_value": MAX_SEQ_LENGTH,
        },
    )

    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except Exception as exc:
        debug_log(
            run_id,
            "H_INIT_EXCEPTION",
            "train.py:trainer_init_exception",
            "SFTTrainer init failed",
            {
                "exception_type": type(exc).__name__,
                "exception_text": str(exc),
            },
        )
        raise

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # --- Train ---
    print("\nStarting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(
        f"  Batch size: {PER_DEVICE_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = "
        f"{PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print()

    trainer.train()

    # --- Save ---
    final_path = f"{OUTPUT_DIR}/final"
    print(f"\nSaving model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nTraining complete!")
    print(f"LoRA adapter saved to: {final_path}")
    print("Run 'python inference.py' to test the model.")


if __name__ == "__main__":
    main()