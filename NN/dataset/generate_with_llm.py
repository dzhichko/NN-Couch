"""
Generate high-quality coaching dialogues using DeepSeek API.

Setup:
    1. Get API key: https://platform.deepseek.com/api_keys
    2. pip install openai
    3. Set environment variable: set DEEPSEEK_API_KEY=your_key_here
    4. Run: python dataset/generate_with_llm.py

Progress is saved incrementally — you can stop and resume safely.
DeepSeek pricing: ~$0.14 per 1M tokens (500 dialogues ≈ $0.10 total).
"""

import json
import os
import sys
import time
import random

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SYSTEM_PROMPT

# === Configuration ===

NUM_DIALOGUES = 500
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "coaching_llm_generated.json")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "generation_progress.json")
MODEL_NAME = "deepseek-chat"        # DeepSeek-V3 — fast and cheap
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 30  # seconds, for rate limit errors

# === Diverse scenario pool ===

SCENARIOS = [
    # Fitness & Health
    {"topic": "похудение", "context": "офисный работник, 30 лет, набрал вес за последние 2 года"},
    {"topic": "бег", "context": "новичок, хочет пробежать первые 5 км"},
    {"topic": "подтягивания", "context": "парень 25 лет, может сделать 3 раза, хочет 20"},
    {"topic": "йога", "context": "девушка с сидячей работой, болит спина"},
    {"topic": "правильное питание", "context": "студент с ограниченным бюджетом"},
    {"topic": "марафон", "context": "бегает 10 км, хочет подготовиться к марафону за полгода"},
    {"topic": "набор мышечной массы", "context": "худощавый парень, никогда не занимался"},
    {"topic": "плавание", "context": "взрослый, который боится воды, но хочет научиться"},
    {"topic": "утренняя зарядка", "context": "хочет выработать привычку, но не может проснуться"},
    {"topic": "отказ от сахара", "context": "сладкоежка, ест десерт каждый день"},

    # Career
    {"topic": "повышение на работе", "context": "работает 3 года, не растёт в должности"},
    {"topic": "смена профессии", "context": "бухгалтер, хочет уйти в IT"},
    {"topic": "фриланс", "context": "дизайнер, хочет уйти из офиса на фриланс"},
    {"topic": "собеседование", "context": "боится собеседований, отказали 5 раз"},
    {"topic": "руководство командой", "context": "впервые стал тимлидом, команда из 4 человек"},
    {"topic": "выгорание на работе", "context": "работает без отпуска 2 года, нет сил"},
    {"topic": "зарплата", "context": "хочет попросить повышение зарплаты на 30%"},
    {"topic": "свой бизнес", "context": "программист, есть идея SaaS-продукта"},
    {"topic": "публичные выступления", "context": "разработчик, нужно выступить на конференции"},
    {"topic": "удалённая работа", "context": "не может сосредоточиться дома, прокрастинирует"},

    # Education & Skills
    {"topic": "английский язык", "context": "уровень A2, нужен B2 для работы за полгода"},
    {"topic": "программирование Python", "context": "полный новичок, гуманитарий"},
    {"topic": "подготовка к экзамену", "context": "студент, сессия через 2 месяца, ничего не учил"},
    {"topic": "чтение книг", "context": "хочет читать 1 книгу в неделю, сейчас 0"},
    {"topic": "музыкальный инструмент", "context": "хочет научиться играть на гитаре с нуля"},
    {"topic": "рисование", "context": "всегда хотел рисовать, но считает что нет таланта"},
    {"topic": "data science", "context": "аналитик, хочет перейти в ML/DS"},
    {"topic": "китайский язык", "context": "интересуется культурой, хочет базовый уровень"},

    # Finance
    {"topic": "накопления", "context": "живёт от зарплаты до зарплаты, нет подушки безопасности"},
    {"topic": "долги", "context": "3 кредита, хочет закрыть все за год"},
    {"topic": "инвестирование", "context": "новичок, есть 50 тысяч, не знает куда вложить"},
    {"topic": "бюджет", "context": "не понимает, куда уходят деньги каждый месяц"},
    {"topic": "накопить на квартиру", "context": "молодая пара, хотят первоначальный взнос"},
    {"topic": "финансовая грамотность", "context": "никогда не вёл бюджет, хочет начать"},

    # Personal development
    {"topic": "прокрастинация", "context": "откладывает всё на последний момент"},
    {"topic": "режим сна", "context": "ложится в 3 ночи, встаёт в 11, хочет наладить"},
    {"topic": "уверенность в себе", "context": "стесняется высказывать мнение, боится конфликтов"},
    {"topic": "медитация", "context": "слышал что полезно, пробовал — не получается"},
    {"topic": "зависимость от телефона", "context": "проводит 6+ часов в соцсетях ежедневно"},
    {"topic": "перфекционизм", "context": "не начинает дела, потому что боится сделать неидеально"},
    {"topic": "тайм-менеджмент", "context": "много задач, не успевает ничего, стресс"},
    {"topic": "дневник", "context": "хочет вести дневник, но забрасывает через 3 дня"},
    {"topic": "утренняя рутина", "context": "хочет продуктивное утро вместо скролла телефона"},

    # Social & Relationships
    {"topic": "новые знакомства", "context": "переехал в новый город, нет друзей"},
    {"topic": "отношения с партнёром", "context": "часто ссорятся, хочет наладить общение"},
    {"topic": "нетворкинг", "context": "интроверт, нужно расширять деловые контакты"},
    {"topic": "границы", "context": "не умеет говорить 'нет', всем помогает в ущерб себе"},
    {"topic": "публичность", "context": "хочет вести блог, но боится осуждения"},
    {"topic": "конфликты", "context": "избегает конфликтов, копит обиды"},
    {"topic": "отношения с родителями", "context": "сложные отношения, хочет наладить"},
]

# === Dialogue types for variety ===

DIALOGUE_TYPES = [
    {
        "type": "goal_setting",
        "instruction": "Пользователь приходит с новой целью. Коуч помогает уточнить цель, задаёт вопросы, и составляет SMART-план.",
        "turns": "6-8",
    },
    {
        "type": "progress_checkin",
        "instruction": "Пользователь возвращается через неделю и рассказывает о прогрессе (успешном или нет). Коуч анализирует, корректирует план.",
        "turns": "4-6",
    },
    {
        "type": "obstacle",
        "instruction": "Пользователь столкнулся с трудностью и хочет бросить. Коуч поддерживает, помогает найти решение и мотивирует продолжать.",
        "turns": "6-8",
    },
    {
        "type": "celebration",
        "instruction": "Пользователь достиг цели или промежуточного результата. Коуч празднует успех и помогает поставить новую цель.",
        "turns": "4-6",
    },
    {
        "type": "exploration",
        "instruction": "Пользователь не знает, чего хочет. Коуч помогает разобраться в себе через вопросы и рефлексию.",
        "turns": "6-10",
    },
    {
        "type": "accountability",
        "instruction": "Пользователь не выполнил план и чувствует вину. Коуч проявляет эмпатию, помогает понять причины и упрощает план.",
        "turns": "6-8",
    },
]

# === Meta-prompt ===

META_PROMPT = """Сгенерируй ОДИН реалистичный диалог между лайф-коучем и клиентом.

СЦЕНАРИЙ:
- Тема: {topic}
- Контекст клиента: {context}
- Тип диалога: {dialogue_type}
- {instruction}

ТРЕБОВАНИЯ К ДИАЛОГУ:
1. Количество реплик (user + assistant): {turns} сообщений
2. Клиент пишет разговорным языком, с ошибками, сокращениями, как в мессенджере
3. Коуч отвечает структурированно, тепло, использует SMART-методику
4. Коуч задаёт уточняющие вопросы, а не сразу даёт советы
5. Коуч предлагает конкретные шаги в конце каждого ответа
6. Диалог должен быть РЕАЛИСТИЧНЫМ — как настоящий разговор
7. Клиент может сомневаться, спорить, задавать вопросы
8. НЕ используй шаблонные фразы типа "Отлично!", "Молодец!" в каждом ответе — варьируй

ФОРМАТ ОТВЕТА — строго JSON, без markdown:
[
  {{"role": "user", "content": "текст клиента"}},
  {{"role": "assistant", "content": "текст коуча"}},
  ...
]

Верни ТОЛЬКО JSON-массив сообщений, без пояснений."""


def create_prompt(scenario: dict, dialogue_type: dict) -> str:
    return META_PROMPT.format(
        topic=scenario["topic"],
        context=scenario["context"],
        dialogue_type=dialogue_type["type"],
        instruction=dialogue_type["instruction"],
        turns=dialogue_type["turns"],
    )


def parse_response(text: str) -> list[dict] | None:
    """Extract JSON array from LLM response."""
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        messages = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                messages = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Validate structure
    if not isinstance(messages, list) or len(messages) < 4:
        return None

    for msg in messages:
        if not isinstance(msg, dict):
            return None
        if "role" not in msg or "content" not in msg:
            return None
        if msg["role"] not in ("user", "assistant"):
            return None

    return messages


def load_progress() -> list[dict]:
    """Load previously generated dialogues for resume support."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(dialogues: list[dict]):
    """Save progress incrementally."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)


def generate_dialogues(api_key: str, num_dialogues: int) -> list[dict]:
    """Generate dialogues using DeepSeek API with retry and resume."""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # Resume from previous progress
    dialogues = load_progress()
    start_from = len(dialogues)

    if start_from > 0:
        print(f"Resuming from {start_from} previously generated dialogues")

    if start_from >= num_dialogues:
        print("Already generated enough dialogues!")
        return dialogues

    failed = 0

    print(f"Generating {num_dialogues - start_from} dialogues with {MODEL_NAME}")
    print()

    for i in range(start_from, num_dialogues):
        scenario = random.choice(SCENARIOS)
        dialogue_type = random.choice(DIALOGUE_TYPES)
        prompt = create_prompt(scenario, dialogue_type)

        status = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=2048,
                    temperature=0.9,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.choices[0].message.content
                messages = parse_response(text)

                if messages:
                    dialogue = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *messages,
                        ]
                    }
                    dialogues.append(dialogue)
                    status = "OK"
                else:
                    failed += 1
                    status = "PARSE_ERROR"
                break  # success or parse error — don't retry

            except RateLimitError:
                wait = RETRY_BASE_DELAY * (attempt + 1)
                status = f"RATE_LIMIT (waiting {wait}s, attempt {attempt + 1}/{MAX_RETRIES})"
                print(f"  [{i + 1}/{num_dialogues}] {status}")
                time.sleep(wait)

            except APIError as e:
                failed += 1
                status = f"API_ERROR: {e}"
                break

        if status is None:
            failed += 1
            status = "MAX_RETRIES_EXCEEDED"

        # Progress
        print(f"  [{i + 1}/{num_dialogues}] "
              f"topic={scenario['topic']:<20s} "
              f"type={dialogue_type['type']:<16s} "
              f"status={status}")

        # Save progress every 10 dialogues
        if len(dialogues) % 10 == 0 and len(dialogues) > start_from:
            save_progress(dialogues)

    # Final save
    save_progress(dialogues)
    print(f"\nDone! Generated: {len(dialogues)}, Failed: {failed}")
    return dialogues


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if not api_key:
        print("=" * 60)
        print("  DeepSeek API Key Setup")
        print("=" * 60)
        print()
        print("1. Go to: https://platform.deepseek.com/api_keys")
        print("2. Create an API key (top up ~$1 — хватит на тысячи диалогов)")
        print("3. Run one of:")
        print()
        print("   Windows CMD:")
        print('   set DEEPSEEK_API_KEY=your_key_here')
        print()
        print("   Windows PowerShell:")
        print('   $env:DEEPSEEK_API_KEY="your_key_here"')
        print()
        print("   Linux/Mac:")
        print('   export DEEPSEEK_API_KEY=your_key_here')
        print()
        print("4. Then run this script again")
        print()

        api_key = input("Or paste your API key here: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return

    # Load existing hand-crafted data
    hand_crafted = []
    hand_crafted_path = os.path.join(SCRIPT_DIR, "coaching_data.json")
    if os.path.exists(hand_crafted_path):
        with open(hand_crafted_path, "r", encoding="utf-8") as f:
            hand_crafted = json.load(f)
        print(f"Loaded {len(hand_crafted)} hand-crafted dialogues")

    # Generate new dialogues
    generated = generate_dialogues(api_key, NUM_DIALOGUES)

    # Merge and shuffle
    all_data = hand_crafted + generated
    random.seed(42)
    random.shuffle(all_data)

    print(f"\nTotal dataset: {len(all_data)} dialogues")
    print(f"  - Hand-crafted: {len(hand_crafted)}")
    print(f"  - LLM-generated: {len(generated)}")

    # Save final dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUTPUT_FILE}")

    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Cleaned up progress file")

    print()
    print("Next steps:")
    print("  1. Run training: python train.py")


if __name__ == "__main__":
    main()
