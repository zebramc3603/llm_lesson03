#!/usr/bin/env python
# Fine-tune an open LLM locally via QLoRA (4-bit) when available; with robust Windows fallbacks.

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--data_path", type=str, default="train.jsonl", help="JSONL file or HF dataset name")
    p.add_argument("--text_field", type=str, default=None, help="If your JSONL has 'text', set this to 'text'")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA into full weights after training")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (Ampere+ GPUs) if available; else fp16 on CUDA")
    return p.parse_args()


def bnb_available() -> bool:
    """Return True only if bitsandbytes is importable AND CUDA is available."""
    try:
        import bitsandbytes as bnb  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def get_bnb_config():
    """Prefer QLoRA (4-bit) if bitsandbytes+CUDA are available; otherwise return None."""
    if not bnb_available():
        return None
    sm_version = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    compute_dtype = torch.bfloat16 if sm_version >= 8 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def find_target_modules_for_llama_like():
    # Works for LLaMA/Mistral/TinyLlama/Qwen2.*-CausalLM naming
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]


def format_sample(example, tokenizer):
    """
    Support two dataset styles:
      1) instruction/input/output fields
      2) a single 'text' field already formatted
    Prefer chat template if available.
    """
    if "text" in example and example["text"] and str(example["text"]).strip():
        return {"text": example["text"]}

    instruction = (example.get("instruction") or "").strip()
    input_txt = (example.get("input") or "").strip()
    output_txt = (example.get("output") or "").strip()

    msgs = [{"role": "user", "content": f"{instruction}\n\n{input_txt}".strip()}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": output_txt}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = (
            "### Instruction:\n" + instruction
            + (f"\n\n### Input:\n{input_txt}" if input_txt else "")
            + "\n\n### Response:\n" + output_txt
        )
    return {"text": text}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adapter_dir = os.path.join(args.output_dir, "adapter")

    print(f"[Config] Loading base model: {args.model_name}")
    quant_config = get_bnb_config()

    # Decide precision flags
    bf16_ok = (
        args.bf16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )
    fp16_ok = torch.cuda.is_available() and not bf16_ok

    # Load model (4-bit if possible)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,  # None -> full precision
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    print(f"[Data] Loading dataset from: {args.data_path}")
    if os.path.isfile(args.data_path):
        ds = load_dataset("json", data_files=args.data_path, split="train")
    else:
        ds = load_dataset(args.data_path, split="train")  # HF Hub dataset name

    if args.text_field:
        # Use provided text field directly
        ds = ds.filter(lambda x: x.get(args.text_field) is not None)
        ds = ds.map(lambda x: {"text": x[args.text_field]})
    else:
        ds = ds.map(lambda x: format_sample(x, tokenizer), remove_columns=ds.column_names)

    # LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=find_target_modules_for_llama_like(),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Optimizer: paged_adamw_8bit needs bitsandbytes. Fallback to adamw_torch if not available.
    optim_name = "paged_adamw_8bit" if bnb_available() else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        # evaluation_strategy="no",
        eval_strategy="no",
        bf16=bf16_ok,
        # fp16=fp16_ok,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        optim=optim_name,
    )

    print("[Train] Starting SFT with LoRA (QLoRA if 4-bit available)…")
    from datasets import Dataset

    def format_dataset(example):
        return {"text": example["text"]}
    ds = ds.map(format_dataset)
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        peft_config=lora_cfg,
        train_dataset=ds,
        # dataset_text_field="text",
        # max_seq_length=args.max_seq_len,
        # packing=True,  # pack multiple samples per sequence for throughput
        args=training_args,
    )
    trainer.train()

    # Save LoRA adapter
    print(f"[Save] Saving LoRA adapter to {adapter_dir}")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Optional: merge LoRA into base model (produces full weights; large!)
    if args.merge_lora:
        print("[Merge] Merging LoRA into base weights (this may take a while)…")
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        # Load base for merge (float16 on CUDA, else float32)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(base, adapter_dir, device_map="auto")
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"[Done] Merged model saved to: {merged_dir}")

    print("[Complete] Training finished.")


if __name__ == "__main__":
    main()
