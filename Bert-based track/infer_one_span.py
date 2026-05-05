import json
import sys
import numpy as np
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoConfig
)
from collections import defaultdict
from transformers import DebertaV2Tokenizer, DebertaTokenizer # 包含 DeBERTa tokenizers


MODEL_ZOO = {
    "distilbert": "../output/span/models/distilbert-base-uncased",
    "bert_base": "../output/span/models/bert-base-uncased",
    "roberta_large": "../output/span/models/roberta-large",
    "deberta_v3_large": "../output/span/models/microsoft-deberta-v3-large"
}

TEST_FILE = "../Data/test_span_rehydrated.jsonl"
SUBMISSION_DIR = "../output/submissions"

BATCH_SIZE = 32
MAX_LEN = 256

# Label definitions 
MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]
LABEL_LIST = ["O"]
for mt in MARKER_TYPES:
    LABEL_LIST.append(f"B-{mt}")
    LABEL_LIST.append(f"I-{mt}")

id2label = {i: l for i, l in enumerate(LABEL_LIST)}
label2id = {l: i for i, l in enumerate(LABEL_LIST)}

print(f"Label Map Restored: {len(LABEL_LIST)} classes defined.")



def load_data(file_path):
    """Load data from JSONL."""
    data = []
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")
                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i}")
    print(f"Loaded {len(data)} samples.")
    return data


def tokenize_for_inference(examples, tokenizer):
    """Preprocessing function."""
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )
    return tokenized_inputs


def extract_bio_spans(prediction_ids, offset_mapping, original_text, id2label):
    """BIO Decoder to convert Token IDs to Spans."""
    extracted_markers = []
    current_type = None
    current_start = None
    current_end = None

    for idx, pred_id in enumerate(prediction_ids):
        offset = offset_mapping[idx]
        if offset is None or (offset[0] == 0 and offset[1] == 0 and idx > 0):
            continue

        label_str = id2label.get(pred_id, "O")

        if label_str == "O":
            prefix = "O"
            tag_type = None
        else:
            prefix, tag_type = label_str.split("-")

        token_start, token_end = offset

        if prefix == "B":
            if current_type is not None:
                extracted_markers.append({
                    "type": current_type,
                    "startIndex": current_start,
                    "endIndex": current_end,
                    "text": original_text[current_start:current_end]
                })
            current_type = tag_type
            current_start = token_start
            current_end = token_end

        elif prefix == "I":
            if current_type == tag_type and current_start is not None:
                current_end = token_end
            else:
                if current_type is not None:
                    extracted_markers.append({
                        "type": current_type,
                        "startIndex": current_start,
                        "endIndex": current_end,
                        "text": original_text[current_start:current_end]
                    })
                current_type = tag_type
                current_start = token_start
                current_end = token_end

        elif prefix == "O":
            if current_type is not None:
                extracted_markers.append({
                    "type": current_type,
                    "startIndex": current_start,
                    "endIndex": current_end,
                    "text": original_text[current_start:current_end]
                })
                current_type = None
                current_start = None
                current_end = None

    if current_type is not None:
        extracted_markers.append({
            "type": current_type,
            "startIndex": current_start,
            "endIndex": current_end,
            "text": original_text[current_start:current_end]
        })

    return extracted_markers


if __name__ == '__main__':

    # 1. Load Test Data
    raw_data = load_data(TEST_FILE)
    dataset = Dataset.from_list(raw_data)

    # 2. Iterate through all models in MODEL_ZOO
    for model_name, model_path in MODEL_ZOO.items():
        print("\n" + "=" * 50)
        print(f"Processing Model: {model_name}")
        print(f"Path: {model_path}")
        print("=" * 50)

        if not os.path.exists(model_path):
            print(f"Error: Path not found. Model parameters may not have been generated. Skip the model {model_name}...")
            continue

        # Define Output Filename
        if not os.path.exists(SUBMISSION_DIR): os.makedirs(SUBMISSION_DIR)
        submission_filename = os.path.join(SUBMISSION_DIR, f"Submission_ner_{model_name}.jsonl")
        
        # Load Model & Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            
        except Exception as e:
            print(f"Error: Failed to load model {model_name}. Reason: {e.__class__.__name__}: {e}")
            print("Skip this model. Please check if the model file is complete.")
            continue

        # Preprocess Data 
        print(f"Tokenizing data for {model_name}...")
        
        tokenized_dataset = dataset.map(
            lambda x: tokenize_for_inference(x, tokenizer),
            batched=True,
            desc=f"Tokenizing for {model_name}",
            remove_columns=[c for c in dataset.column_names if c not in ["text", "_id", "conspiracy"]]
        )

        # Run Inference
        print(f"Running Inference for {model_name}...")
        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_infer_{model_name}",
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="none"
            ),
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        predictions_output = trainer.predict(tokenized_dataset)
        logits = predictions_output.predictions
        predictions_ids = np.argmax(logits, axis=2)

        # Reconstruct Spans
        print("Reconstructing spans...")
        final_lines = []

        for i, pred_ids in enumerate(predictions_ids):
            original_item = raw_data[i]
            offset_mapping = tokenized_dataset[i]["offset_mapping"] 
            text = original_item["text"]

            predicted_markers = extract_bio_spans(pred_ids, offset_mapping, text, id2label)

            submission_obj = {
                "_id": original_item["_id"],
                "conspiracy": original_item["conspiracy"],
                "markers": predicted_markers
            }
            final_lines.append(json.dumps(submission_obj))

        # Save Submission File
        with open(submission_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_lines) + '\n')

        print(f"✅ Submission generated successfully: {submission_filename}")

        # Clean up memory
        del model, trainer, tokenizer
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("All models have completed prediction!")
    print("=" * 50)