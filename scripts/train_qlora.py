import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

def main():
    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir="finetune/out",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
    )

    def format_example(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        formatting_func=format_example,
        max_seq_length=1024,  # 6GB VRAM için gerekirse 768 yaparız
        packing=False,
    )

    trainer.train()
    trainer.save_model("finetune/out/lora_adapter")
    tokenizer.save_pretrained("finetune/out/lora_adapter")
    print("Saved LoRA adapter -> finetune/out/lora_adapter")

if __name__ == "__main__":
    main()
