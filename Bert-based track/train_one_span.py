import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch 

import json
import numpy as np
import shutil
import evaluate 
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)



TRAIN_FILE = "../Data/train_cleaned_final.jsonl"
OUTPUT_ROOT = "../output/span"
MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "models")


# List of models to train and compare
MODEL_ZOO = [
    "distilbert-base-uncased", 
    "bert-base-uncased", 
    "roberta-large", 
    "microsoft/deberta-v3-large"
    ]

# Training Hyperparameters
BATCH_SIZE = 32 
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10 
MAX_LEN = 256  

# Label Definitions (BIO Scheme)
MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]
LABEL_LIST = ["O"]
for mt in MARKER_TYPES:
    LABEL_LIST.append(f"B-{mt}")
    LABEL_LIST.append(f"I-{mt}")

# Create Label Mappings
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for l, i in label2id.items()}

print(f"Label System: {len(LABEL_LIST)} classes defined.")


def load_data(file_path):
    data = []
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Ensure valid data
                if "text" in item and "markers" in item:
                    data.append(item)
            except json.JSONDecodeError:
                pass
    return data


# Tokenization Logic 
def create_tokenize_fn(tokenizer):
    """Initializes tokenization and aligns BIO labels with token offsets."""
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=True, 
        )
        labels = []
        for i, markers in enumerate(examples["markers"]):
            doc_offset = tokenized_inputs["offset_mapping"][i]
            label_ids = [-100] * len(doc_offset)

            text_len = len(examples["text"][i])
            char_labels = ["O"] * text_len

            # 1. Map markers to character-level BIO tags
            for marker in markers:
                m_type = marker['type']
                start, end = marker['startIndex'], marker['endIndex']
                if start >= text_len: continue
                end = min(end, text_len)

                if end > start:
                    char_labels[start] = f"B-{m_type}"
                    for k in range(start + 1, end):
                        char_labels[k] = f"I-{m_type}"

            # 2. Align character tags to token tags
            for idx, (start, end) in enumerate(doc_offset):
                # 跳过特殊 tokens (offset 0,0) 或 padding
                if start == end:
                    continue

                if start < text_len:
                    label_ids[idx] = label2id.get(char_labels[start], 0)
                else:
                    label_ids[idx] = -100  # 忽略 padding/truncation 部分
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels

seqeval_metric = evaluate.load("seqeval")


def compute_metrics(p):
    """Computes overall and per-category F1 scores using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs back to Tags
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    
    metrics_to_return = {
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"]
    }

    for marker in MARKER_TYPES:
        key_name = f"f1_{marker.lower()}"
        if marker in results:
            metrics_to_return[key_name] = results[marker]["f1"]
        else:
            metrics_to_return[key_name] = 0.0
            
    return metrics_to_return



def validate_alignment(dataset, tokenizer, num_samples=2): 
    """Debugging utility to verify character-to-token label alignment."""
    print("\n" + "="*20 + " DATA ALIGNMENT VALIDATION START " + "="*20)
    
    tokenize_fn = create_tokenize_fn(tokenizer)
    samples_to_validate = dataset[:num_samples] 
    tokenized_output = tokenize_fn(samples_to_validate)
    
    for i in range(num_samples):
        print(f"\n--- SAMPLE {i+1} ---")
        
        raw_text = samples_to_validate["text"][i] 
        print(f"Original Text: {raw_text[:100]}...")
        
        raw_markers = samples_to_validate["markers"][i]
        print(f"Raw Markers: {raw_markers}")
        
        input_ids = tokenized_output["input_ids"][i]
        label_ids = tokenized_output["labels"][i]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        aligned_labels = [id2label[lid] if lid != -100 else "IGNORE" for lid in label_ids]
        
        print("\n| Index | Token (Raw) | Aligned Label |")
        print("|-------|-------------|---------------|")
        
        for idx, (token, label) in enumerate(zip(tokens, aligned_labels)):
            if label != "IGNORE" or token in tokenizer.all_special_tokens:
                 print(f"| {idx:<5} | {token:<11} | {label:<13} |")
            
    print("="*20 + " DATA ALIGNMENT VALIDATION END " + "="*20)


if __name__ == "__main__":
    # 1. Prepare Data 
    raw_data = load_data(TRAIN_FILE)
    raw_dataset = Dataset.from_list(raw_data)
    # 90% Train, 10% Validation
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Data Prepared: Train {len(split_dataset['train'])}, Test {len(split_dataset['test'])}")

    results_summary = []

    # 2. Iterate through each model in the zoo
    for model_name in MODEL_ZOO:
        print("\n" + "=" * 50)
        print(f"Training Model: {model_name}")
        print("=" * 50)
        
        model_f1 = "N/A" 
        status = "Failed to Load"

        try:
            # A. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True 
            )
            
            if model_name == MODEL_ZOO[0]: 
                print("\n>>> Running Data Alignment Validation (Checking Labels) <<<")
                validate_alignment(split_dataset['train'], tokenizer, num_samples=2)
            
            # B. Process Data 
            tokenize_fn = create_tokenize_fn(tokenizer)
            print(f"Tokenizing data for {model_name}...")
            tokenized_datasets = split_dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)

            # C. Initialize Model
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(LABEL_LIST),
                id2label=id2label,
                label2id=label2id
            )

            # D. Setup Paths
            safe_name = model_name.replace("/", "-")
            ckpt_dir = os.path.join(OUTPUT_ROOT, "checkpoints", safe_name)
            final_save_path = os.path.join(OUTPUT_ROOT, "best_models", safe_name)

            # E. Training Arguments 
            args = TrainingArguments(
                output_dir=ckpt_dir,
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=NUM_EPOCHS,
                weight_decay=0.01,

                save_total_limit=1,  

                logging_steps=50,
                report_to="none",
                fp16=True,  

                evaluation_strategy="epoch",       
                save_strategy="epoch",             
                load_best_model_at_end=True,       
                metric_for_best_model="f1",        
                greater_is_better=True,            
            )

            trainer = Trainer(
                model=model,
                args=args, 
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                tokenizer=tokenizer,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics,
            )

            # F. Train
            trainer.train()
            status = "Completed" # 训练成功完成

            # G. Evaluate and Save Final "Best" Model
            print(f"Evaluating best version of {model_name}...")
            metrics = trainer.evaluate()
            
            best_f1 = metrics["eval_f1"]
            model_f1 = f"{best_f1:.4f}"

            print(f"Saving best model to: {final_save_path}")
            trainer.save_model(final_save_path)
            tokenizer.save_pretrained(final_save_path)

            print(f"{model_name} Finished! Best F1: {model_f1}")
            for marker in MARKER_TYPES:
                 f1_key = f"eval_f1_{marker.lower()}"
                 if f1_key in metrics:
                     print(f" {marker} F1: {metrics[f1_key]:.4f}")

            # Cleanup GPU Memory
            del model, trainer
            torch.cuda.empty_cache()

        except Exception as e:

            status = f"Failed ({e.__class__.__name__})"
            print(f"\nFATAL ERROR for {model_name}: {e}")
            print(f"Skipping remaining models in this loop iteration.")
            
        finally:
            # Log Result
            results_summary.append({
                "Model": model_name,
                "Best F1": model_f1,
                "Saved At": "N/A" if model_f1 == "N/A" else final_save_path,
                "Status": status
            })


    # 3. Final Leaderboard
    print("\n" + "#" * 70)
    print("Training Summary & Leaderboard (Sorted by F1)")
    print("#" * 70)
    print(f"{'Model':<30} | {'Best F1':<10} | {'Status':<15} | {'Saved Path'}")
    print("-" * 70)


    def sort_key(res):
        try:
            return float(res["Best F1"])
        except ValueError:
            return -1.0 

    results_summary.sort(key=sort_key, reverse=True)

    for res in results_summary:
        print(f"{res['Model']:<30} | {res['Best F1']:<10} | {res['Status']:<15} | {res['Saved At']}")
    print("#" * 70)