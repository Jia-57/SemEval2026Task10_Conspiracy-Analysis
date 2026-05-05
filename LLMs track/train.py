import argparse
import json
import torch
import os
import traceback
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from utils1 import build_prompt

def manual_concat(messages):
    """Manually constructs a text prompt from chat messages for models without a built-in chat template."""
    human_text = ""
    assistant_text = ""
    for m in messages:
        if m['role'] in ['system', 'user']:
            human_text += f"{m['content']}\n"
        elif m['role'] == 'assistant':
            assistant_text = m['content']
    return f"Human:\n{human_text}Assistant:\n{assistant_text}"

def process_data_to_text(examples, tokenizer):
    """Prepares raw data samples into formatted strings suitable for SFT model training."""
    output_texts = []
    for text, label, markers in zip(examples['text'], examples['conspiracy'], examples['markers']):
        messages = build_prompt("C", text)
        response_json = {
            "reasoning": "Analysis based on psycholinguistic markers.",
            "conspiracy": label,
            "markers": [{"text": m["text"], "type": m["type"]} for m in markers]
        }
        assistant_content = json.dumps(response_json, ensure_ascii=False)
        messages.append({"role": "assistant", "content": assistant_content})
        
        if tokenizer.chat_template is not None:
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                prompt = manual_concat(messages)
        else:
            prompt = manual_concat(messages)
            
        output_texts.append(prompt)
    return {"text": output_texts}

def train(args):
    """Main execution function for LoRA fine-tuning using SFTTrainer with optimized GPU settings[cite: 22, 42]."""
    print(f"DEBUG: Starting train process for {args.model_path}")
    try:
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # Use right-side padding to prevent deadlocks in FP16 training
        tokenizer.padding_side = 'right'
        tokenizer.model_max_length = 1024
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model (FP16)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Disable cache and enable checkpointing to optimize memory and stability 
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        print(f"Processing data from {args.train_file}...")
        data = []
        with open(args.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        raw_dataset = Dataset.from_list(data)

        dataset = raw_dataset.map(
            process_data_to_text,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=raw_dataset.column_names
        )
        
        print(f"Dataset processed. Sample (Length {len(dataset[0]['text'])} chars).")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,   
            learning_rate=2e-4,
            lr_scheduler_type="cosine",      
            num_train_epochs=1,              
            warmup_ratio=0.03,               
            logging_steps=1,                 
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            fp16=True,                       
            optim="paged_adamw_32bit",       
            report_to="none",
            gradient_checkpointing=True,     
            remove_unused_columns=True 
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1024,             
            args=training_args,
            tokenizer=tokenizer 
        )

        print(">>> Trainer initialized. Starting training...")
        trainer.train()
        
        print(f"Saving adapter to {args.output_dir}")
        trainer.save_model(args.output_dir)
        print("SUCCESS!!")

    except Exception as e:
        print("\n ERROR DETECTED!")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="train_cleaned_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="lora_output")
    args = parser.parse_args()
    train(args)


