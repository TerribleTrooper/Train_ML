#!/usr/bin/env python3
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
OUTPUT_DIR = "./output_phi3_lora"
DATA_PATH = "./age_dataset.jsonl"

SYSTEM_PROMPT = (
    "Твоя задача — анализировать текст и присваивать ему возрастной рейтинг.\n"
    "ПРОЦЕДУРА АНАЛИЗА:\n"
    "1. Внимательно прочитай текст.\n"
    "2. Ищи конкретные нарушения по категориям:\n"
    "   - НАСИЛИЕ: оружие, драки, убийства, кровь, жестокость.\n"
    "   - НЕНОРМАТИВНАЯ ЛЕКСИКА: мат, ругательства, оскорбления.\n"
    "   - СЕКСУАЛЬНЫЙ КОНТЕНТ: интимные сцены, обнажение, эротика.\n"
    "   - АЛКОГОЛЬ/НАРКОТИКИ: только если в тексте ЯВНО описано употребление или пропаганда "
    "конкретных веществ (например: пьёт, напился, курит, нюхает, колется, принял таблетку, "
    "под кайфом, торгует наркотиками и т.п.).\n"
    "   - ОТДЕЛЯЙ имена собственные и фамилии от веществ. Если слово похоже на наркотик, "
    "но используется как имя персонажа (например, «Белладонна Тукк»), НЕ считай это "
    "категорией АЛКОГОЛЬ/НАРКОТИКИ.\n"
    "   - ПУГАЮЩИЙ КОНТЕНТ: ужасы, психологическое давление.\n\n"
    "ВОЗРАСТНЫЕ КАТЕГОРИИ:\n"
    "0+  - Полностью безопасно, детский контент.\n"
    "6+  - Мягкие условности (персонажи в опасности без деталей).\n"
    "12+ - Умеренное насилие без крови, лёгкий испуг.\n"
    "16+ - Явное насилие, алкоголь/табак, сексуальные отсылки.\n"
    "18+ - Жестокость, откровенный секс, наркотики, тяжёлые темы.\n\n"
    "ПРАВИЛА ОТВЕТА:\n"
    "- Анализируй КОНКРЕТНО этот текст, а не шаблонно.\n"
    "- В поле \"why\" укажи реальную причину на русском.\n"
    "- В поле \"label\" укажи основную категорию нарушения.\n"
    "- Если нарушений нет — ставь 0+.\n"
    "- Будь строгим, но справедливым.\n\n"
    "- При сомнении (когда слово может быть и именем, и веществом) — выбирай более мягкий "
    "вариант и НЕ ставь категорию АЛКОГОЛЬ/НАРКОТИКИ без явного употребления.\n"
    "ФОРМАТ ОТВЕТА:\n"
    "- Выводи СТРОГО ОДИН JSON-объект.\n"
    "- Никакого дополнительного текста ДО или ПОСЛЕ JSON.\n"
    "- Используй только двойные кавычки для ключей и значений.\n"
    "Только один объект формата:\n"
    '{\"rating\": \"...\", \"why\": \"...\", \"label\": \"...\"}\n'
)

def build_prompt(input_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\nВот текст для анализа:\n"
        + input_text
        + "\n\nОтветь ОДНИМ JSON-объектом:"
    )

def formatting_func(example):
    """
    Принимает ОДИН пример: {"input": ..., "output": ...}
    и возвращает одну строку.
    """
    prompt = build_prompt(example["input"])
    full = prompt + "\n" + example["output"]
    return full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Использую устройство:", device)

    # Загружаем датасет
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    model.to(device)
    model.config.use_cache = False  # важно для тренировки с gradient checkpointing / LoRA

    # LoRA-конфиг (если что-то упадёт, можно подправить target_modules под реальные имена слоёв)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1 if device == "cpu" else 4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=(device == "cuda"),
        fp16=False,
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,  # вместо tokenizer=tokenizer
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Готово, адаптер сохранён в:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
