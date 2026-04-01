# NN-Couch — Нейрокоуч на базе Qwen

Файн-тюнинг модели **Qwen/Qwen2.5-7B-Instruct** под задачу лайф-коучинга с использованием **QLoRA** (4-bit квантизация + LoRA адаптеры).

---

## Установка
**Важно установить версию с CUDA, без нее обучение займет недели, а то и больше** 
```bash

# Установить зависимости
pip install -r requirements.txt

# Установить PyTorch с поддержкой CUDA\
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Проверить, что GPU видна:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Ожидаемый вывод:
```
CUDA: True
GPU: NVIDIA GeForce RTX XXXX
```

---

## Структура проекта

```
NN/
├── config.py                        # Все гиперпараметры и настройки
├── train.py                         # Обучение модели (QLoRA)
├── inference.py                     # Интерактивный чат с моделью
├── requirements.txt                 # Зависимости
└── dataset/
    ├── coaching_data.json           # Ручные диалоги (базовый датасет)
    ├── coaching_llm_generated.json  # Датасет, сгенерированный через LLM
    ├── prepare_dataset.py           # Генератор шаблонных диалогов (устаревший)
    └── generate_with_llm.py         # Генератор диалогов через DeepSeek API
```

---

## Шаг 1 — Генерация датасета

Для обучения используется датасет из реалистичных диалогов коуча с клиентом.
Скрипт `generate_with_llm.py` генерирует их через **DeepSeek API** (~$0.10 за 500 диалогов).

### Получить API-ключ DeepSeek:
1. Обратиться к tg:@pumpu1

### Запустить генерацию:

```bash
python dataset/generate_with_llm.py
```
Прогресс сохраняется каждые 10 диалогов в `generation_progress.json`.
Если скрипт прервётся — при повторном запуске продолжит с того места.

По умолчанию генерируется **500 диалогов** > часа.
Количество можно изменить в `generate_with_llm.py`:
```python
NUM_DIALOGUES = 500
```

---

## Шаг 2 — Настройка (опционально)

Все параметры обучения находятся в `config.py`:
Надо будет переделать, чтобы они задавались динамически через настройки окружения, но пока так

```python
# Модель
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# LoRA
LORA_R = 8              # Ранг адаптера (меньше = меньше переобучение)
LORA_ALPHA = 16         # Масштаб LoRA

# Обучение
NUM_EPOCHS = 2          # Количество эпох
LEARNING_RATE = 5e-5    # Скорость обучения
MAX_SEQ_LENGTH = 512    # Максимальная длина последовательности
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8   # Эффективный batch = 1 × 8 = 8
```

## Шаг 3 — Обучение

```bash
python train.py
```

Ожидаемый вывод:
```
============================================================
  DQwen/Qwen2.5-7B-Instruct
============================================================

Loading tokenizer: Qwen/Qwen2.5-7B-Instruct
Loading model: Qwen/Qwen2.5-7B-Instruct (4-bit)
Model loaded. Parameters: 1,781,590,016
Dataset size: 521 examples
Initializing SFTTrainer...
Trainable parameters: 5,505,024 / 1,781,590,016 (0.31%)

Starting training...
  Epochs: 2
  Batch size: 1 x 8 = 8
  Learning rate: 5e-05
  Max sequence length: 512

  0%|          | 0/128 [00:00<?, ?it/s]
```

**Первый запуск:** модель (~6 GB) скачается автоматически с HuggingFace.

**Время обучения модели** Более часа, с новой генерацией диалогов будет больше. Еще не тестил

По окончании файн-тюненная модель сохраняется в `./output/final/`.

---

## Шаг 4 — Запуск чата

### Режим файн-тюненной модели (после обучения):

```bash
python inference.py
```

### Режим базовой модели (для сравнения, без обучения):

```bash
python inference.py --base
```

### Указать путь к адаптеру вручную:

```bash
python inference.py --adapter ./output/checkpoint-100
```

### Пример диалога:

```
============================================================
  Online Coach - Interactive Chat
  Commands: /reset (new conversation), /quit (exit)
============================================================

You: Хочу начать бегать, но каждый раз бросаю через неделю

Coach: Понимаю — это очень частая ситуация, и важно разобраться,
почему так происходит. Ответь на пару вопросов:
1. Когда ты обычно бегаешь — утром, вечером?
2. Какой темп и дистанция — сразу берёшь много?
3. Что именно происходит через неделю — физически тяжело
   или просто пропадает мотивация?
```

### Команды в чате:

| Команда | Действие |
|---------|----------|
| `/reset` | Начать новый разговор (очистить историю) |
| `/quit`  | Выйти из чата |

---

## Полный цикл (быстрый старт)

```bash
# 1. Установить зависимости
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install openai

# 2. Сгенерировать датасет
set DEEPSEEK_API_KEY=your_key_here
python dataset/generate_with_llm.py

# 3. Обучить модель
python train.py

# 4. Запустить чат
python inference.py
```

---

## Возможные ошибки

**`CUDA: False` — GPU не видна**
```bash
# Переустановить PyTorch с поддержкой CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**`CUDA out of memory`**
```python
# config.py — уменьшить нагрузку:
PER_DEVICE_BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256

```
