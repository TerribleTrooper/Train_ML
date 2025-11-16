# train_phi3_age_explain.py
# Дообучение Phi-3 под классификацию возрастного рейтинга + объяснение + однословная категория (RU).
# Профиль для CPU/Mac (MPS). 4-bit/CUDA не требуются.

import os
import json
import argparse
import torch
from typing import Dict, Any
from packaging import version
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# --- совместимость некоторых сборок PyTorch на macOS ---
if hasattr(torch.backends, "mps") and not hasattr(torch.backends.mps, "is_macos_or_newer"):
    if hasattr(torch.backends.mps, "is_macos13_or_newer"):
        torch.backends.mps.is_macos_or_newer = torch.backends.mps.is_macos13_or_newer

DEFAULT_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

# Допустимые ярлыки (одно слово в выводе)
ALLOWED_LABELS = {
    "насилие",
    "ненормативная_лексика",
    "эротика",
    "алкоголь_наркотики",
    "пугающие_сцены",
    "другое",
}

SYSTEM_INSTRUCTION = (
    "Вы — эксперт по возрастной классификации РФ (436-ФЗ). "
    "Проанализируйте русскоязычное ПРЕДЛОЖЕНИЕ и верните СТРОГО JSON:\n"
    "{\"rating\":\"0+|6+|12+|16+|18+\","
    "\"why\":\"короткое объяснение на русском\","
    "\"label\":\"насилие|ненормативная_лексика|эротика|алкоголь_наркотики|пугающие_сцены|другое\"}\n"
    "Никаких комментариев вне JSON."
)

USER_TEMPLATE = 'Текст: "{text}"\nВерните только JSON без пояснений.'

def _norm_label(val: str) -> str:
    """Нормализует произвольную строку метки к допустимому единственному слову."""
    if not val:
        return "другое"
    v = val.strip().lower()
    repl = {
        "ненормативная лексика": "ненормативная_лексика",
        "мат": "ненормативная_лексика",
        "брань": "ненормативная_лексика",
        "препараты": "алкоголь_наркотики",
        "наркотики": "алкоголь_наркотики",
        "алкоголь": "алкоголь_наркотики",
        "страшно": "пугающие_сцены",
        "страшные сцены": "пугающие_сцены",
        "ужасы": "пугающие_сцены",
        "секс": "эротика",
        "эротический контент": "эротика",
        "эротика ": "эротика",
        "насилие ": "насилие",
    }
    if v in repl:
        v = repl[v]
    v = v.replace(" ", "_")
    return v if v in ALLOWED_LABELS else "другое"

def row_to_text(ex: Dict[str, Any]) -> str:
    """
    Формат строки train.jsonl:
    {
      "text":   "... одно предложение ...",
      "rating": "12+",
      "why":    "краткое объяснение",
      "label":  "насилие|ненормативная_лексика|эротика|алкоголь_наркотики|пугающие_сцены|другое"
                (если отсутствует — нормализуем в 'другое')
    }
    """
    text = str(ex["text"]).strip()
    rating = ex.get("rating", "0+")
    why = str(ex.get("why", "")).strip()
    label = _norm_label(str(ex.get("label", "")).strip())

    gold = json.dumps(
        {"rating": rating, "why": why, "label": label},
        ensure_ascii=False
    )

    return (
        f"<|system|>\n{SYSTEM_INSTRUCTION}\n"
        f"<|user|>\n{USER_TEMPLATE.format(text=text)}\n"
        f"<|assistant|>\n{gold}"
    )

def guess_lora_targets(model) -> list:
    # Для Phi-3 обычно есть qkv_proj и o_proj.
    names = [n for n, _ in model.named_modules()]
    if any("qkv_proj" in n for n in names) and any("o_proj" in n for n in names):
        return ["qkv_proj", "o_proj"]

    # Фолбэк — любые attention-проекции
    import torch.nn as nn
    proj_like = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and ("attn" in n.lower() or "attention" in n.lower()):
            leaf = n.split(".")[-1]
            if "proj" in leaf.lower():
                proj_like.add(leaf)
    return sorted(proj_like) if proj_like else ["qkv_proj", "o_proj"]

def main():
    ap = argparse.ArgumentParser("Fine-tune Phi-3 for RU age rating + explanation + label")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--train", required=True, help="path to train.jsonl")
    ap.add_argument("--eval", default=None, help="optional dev.jsonl")
    ap.add_argument("--out", required=True, help="output dir for adapter/tokenizer")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = ap.parse_args()

    use_mps = (args.device == "mps") and torch.backends.mps.is_available()
    torch_ver_ok = version.parse(torch.__version__) >= version.parse("2.5.0")
    if use_mps:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        if not torch_ver_ok:
            print(f"[WARN] torch {torch.__version__} < 2.5.0 — fp16 на MPS недоступен (использую float32).")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.model_max_length = args.max_len

    # грузим базовую модель на CPU (экономит VRAM при старте)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    targets = guess_lora_targets(model)
    peft_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.1,
        target_modules=targets, bias="none", task_type="CAUSAL_LM"
    )
    print("LoRA target_modules:", targets)

    train_ds = load_dataset("json", data_files=args.train, split="train")
    train_ds = train_ds.map(lambda ex: {"text": row_to_text(ex)}, num_proc=2)

    do_eval = bool(args.eval)
    eval_ds = None
    if do_eval:
        eval_ds = load_dataset("json", data_files=args.eval, split="train")
        eval_ds = eval_ds.map(lambda ex: {"text": row_to_text(ex)}, num_proc=2)

    sft_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, args.grad_accum),
        learning_rate=args.lr,
        dataset_text_field="text",
        packing=False,
        logging_steps=50,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=300,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        bf16=False,
        fp16=use_mps and torch_ver_ok,
        use_cpu=not use_mps,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        peft_config=peft_cfg,  # ок для новых TRL; если будет ошибка — просто убери этот аргумент
    )

    try:
        last_ckpt = get_last_checkpoint(args.out)
        if last_ckpt:
            print(f"[resume] Resuming from {last_ckpt}")
        try:
            trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Сохраняю прогресс...")
            trainer.save_state()
            trainer.save_model(args.out)
            raise

        trainer.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Сохраняю прогресс...")
        trainer.save_state()
        trainer.save_model(args.out)
        raise

    trainer.save_model()
    tok.save_pretrained(args.out)
    print(f"✅ Готово. Адаптер/токенайзер сохранены в: {args.out}")

if __name__ == "__main__":
    main()
