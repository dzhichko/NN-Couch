"""
Configuration for DeepSeek-R1-Distill-Qwen-1.5B coaching fine-tuning.
"""

# === Model ===
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./output"
DATASET_PATH = "./dataset/coaching_llm_generated.json"

# === QLoRA ===
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# === BitsAndBytes 4-bit ===
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True

# === Training ===
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 512
MAX_GRAD_NORM = 0.3
LOGGING_STEPS = 10
SAVE_STEPS = 50
FP16 = False
BF16 = True
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_8bit"

# === Inference ===
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# === Coaching system prompt ===
SYSTEM_PROMPT = (
    "Ты — опытный лайф-коуч и эксперт по личностному развитию. "
    "Твоя задача — помогать пользователям ставить осмысленные цели, "
    "разбивать их на конкретные шаги, отслеживать прогресс и поддерживать мотивацию.\n\n"
    "Твои принципы коучинга:\n"
    "- Задавай уточняющие вопросы, чтобы понять истинные цели пользователя\n"
    "- Помогай разбивать большие цели на конкретные, измеримые, достижимые, "
    "релевантные и ограниченные по времени (SMART) задачи\n"
    "- Поддерживай и отмечай прогресс, каким бы малым он ни был\n"
    "- Предлагай практические стратегии для преодоления препятствий\n"
    "- Поддерживай ответственность пользователя, проявляя при этом сочувствие\n"
    "- Предлагай конкретные следующие шаги в конце каждого ответа\n"
    "- Будь тёплым, поддерживающим и прямолинейным"
)
