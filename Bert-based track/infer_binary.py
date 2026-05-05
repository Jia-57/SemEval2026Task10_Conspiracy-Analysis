import json
import sys
import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DebertaV2Tokenizer
)


TRAINING_OUTPUT_ROOT = "../output/binary"
TEST_FILE = "../Data/test_binary_rehydrated.jsonl"
SUBMISSION_DIR = "../output/submissions"

# Model List 
MODEL_ZOO = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-large",
    "microsoft/deberta-v3-large"
]

BATCH_SIZE = 32
MAX_LEN = 256
LABEL_MAP = {0: "No", 1: "Yes"}



class SimpleHybridModel(nn.Module):
    def __init__(self, base_model, num_labels, hidden_size, num_subreddits, num_id_types, num_annotators):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        self.subreddit_embedding = nn.Embedding(num_subreddits, 32)
        self.id_type_embedding = nn.Embedding(num_id_types, 8)
        self.annotator_embedding = nn.Embedding(num_annotators, 8)

        total_discrete_dim = 32 + 8 + 8
        combined_dim = hidden_size + total_discrete_dim

        self.fusion_norm = nn.LayerNorm(combined_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, subreddit_ids, id_type_ids, annotator_ids, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = outputs.last_hidden_state[:, 0, :]

        sub_embed = self.subreddit_embedding(subreddit_ids.long())
        id_embed = self.id_type_embedding(id_type_ids.long())
        annotator_embed = self.annotator_embedding(annotator_ids.long())

        combined = torch.cat((text_embed, sub_embed, id_embed, annotator_embed), dim=1)
        combined = self.fusion_norm(combined)

        logits = self.classifier(combined)
        return logits


class InferenceDataset(TorchDataset):
    """Dataset wrapper for tokenizing text and mapping metadata for inference."""
    def __init__(self, data, tokenizer, encoders, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.encoders = encoders
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        def safe_transform(key, val):
            if val in self.encoders[key].classes_:
                return self.encoders[key].transform([val])[0]
            else:
                if "unknown" in self.encoders[key].classes_:
                    return self.encoders[key].transform(["unknown"])[0]
                return 0

        return {
            "unique_sample_id": item['_id'],
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "subreddit_ids": torch.tensor(safe_transform('sub', item['sub_raw']), dtype=torch.long),
            "id_type_ids": torch.tensor(safe_transform('id', item['id_raw']), dtype=torch.long),
            "annotator_ids": torch.tensor(safe_transform('anno', item['anno_raw']), dtype=torch.long),
        }


def load_raw_data(file_path):
    """Parses the test JSONL file into a list of dictionaries."""
    print(f"Reading raw data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                raw_id = item.get('id', '')
                id_prefix = raw_id.split('_')[0] if '_' in raw_id else 'unknown'

                data.append({
                    "_id": item.get("_id", item.get("id", f"line_{i}")),
                    "text": item.get("text", ""),
                    "sub_raw": item.get("subreddit", "unknown"),
                    "id_raw": id_prefix,
                    "anno_raw": item.get("annotator", "unknown")
                })
            except json.JSONDecodeError:
                pass
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Common Resources (Encoders & Raw Data) ---
    encoder_path = os.path.join(TRAINING_OUTPUT_ROOT, "encoders.joblib")
    if not os.path.exists(encoder_path):
        print(f"Error: encoders.joblib not found at {encoder_path}")
        sys.exit(1)

    print(f"Loading shared encoders...")
    encoders = joblib.load(encoder_path)

    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file {TEST_FILE} not found.")
        sys.exit(1)

    raw_data = load_raw_data(TEST_FILE)
    print(f"Total samples to predict: {len(raw_data)}")

    # --- 2. Iterate Through Model Zoo ---
    for model_name in MODEL_ZOO:
        safe_name = model_name.replace("/", "-")
        model_dir = os.path.join(TRAINING_OUTPUT_ROOT, "saved_models", safe_name)
        
        if not os.path.exists(SUBMISSION_DIR): os.makedirs(SUBMISSION_DIR)
        submission_file = os.path.join(SUBMISSION_DIR, f"submission_binary_{safe_name}.jsonl")

        print(f"\n{'=' * 60}")
        print(f"Processing Model: {model_name}")
        print(f"Looking for weights in: {model_dir}")
        print(f"{'=' * 60}")

        # Check if model exists
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            print(f"  Skipping {model_name}: pytorch_model.bin not found (Maybe training failed?).")
            continue

        try:
            # A. Load Tokenizer (Deberta check matches training code)
            print("Loading tokenizer...")
            if "deberta-v3" in model_name:
                tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # B. Load Config & Model Structure
            print("Loading config and model...")
            config = AutoConfig.from_pretrained(model_dir)
            hidden_size = getattr(config, "hidden_size", 768)

            base_model_obj = AutoModel.from_config(config)

            model = SimpleHybridModel(
                base_model=base_model_obj,
                num_labels=2,
                hidden_size=hidden_size,
                num_subreddits=len(encoders['sub'].classes_),
                num_id_types=len(encoders['id'].classes_),
                num_annotators=len(encoders['anno'].classes_)
            )

            # C. Load Weights
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            # D. Prepare Data (Tokenization happens here specific to this model)
            dataset = InferenceDataset(raw_data, tokenizer, encoders, max_len=MAX_LEN)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            # E. Inference
            results = []
            print(f"Running inference...")

            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    sub_ids = batch['subreddit_ids'].to(device)
                    id_type_ids = batch['id_type_ids'].to(device)
                    anno_ids = batch['annotator_ids'].to(device)

                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        subreddit_ids=sub_ids,
                        id_type_ids=id_type_ids,
                        annotator_ids=anno_ids
                    )

                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                    sample_ids = batch['unique_sample_id']
                    for uid, pred in zip(sample_ids, preds):
                        results.append({
                            "_id": uid,
                            "conspiracy": LABEL_MAP[pred]
                        })

            # F. Save Individual Submission File
            print(f"Saving {len(results)} results to {submission_file}...")
            with open(submission_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")

            print(f"{model_name} Done!")

            # Cleanup to save memory
            del model, base_model_obj, tokenizer, dataset, dataloader
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\nAll models processed.")