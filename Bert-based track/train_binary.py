import os
import sys
import shutil
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import evaluate

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
    DebertaV2Tokenizer,
    EvalPrediction
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths 
TRAIN_FILE = "../Data/train_cleaned_final.jsonl"
OUTPUT_ROOT = "../output/binary"
MODEL_SAVE_PATH = os.path.join(OUTPUT_ROOT, "models")
TEST_SIZE = 0.1

# Model List 
MODEL_ZOO = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-large",
    "microsoft/deberta-v3-large"
]

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LEN = 256


class SimpleHybridModel(nn.Module):
    """
    Standard hybrid architecture: Transformer backbone + Categorical Embeddings.
    """
    def __init__(self, base_model, num_labels, hidden_size, num_subreddits, num_id_types, num_annotators):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        # Discrete feature embedding layer
        self.subreddit_embedding = nn.Embedding(num_subreddits, 32)
        self.id_type_embedding = nn.Embedding(num_id_types, 8)
        self.annotator_embedding = nn.Embedding(num_annotators, 8)

        # The dimensions after concating
        total_discrete_dim = 32 + 8 + 8
        combined_dim = hidden_size + total_discrete_dim

        self.fusion_norm = nn.LayerNorm(combined_dim)

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, num_labels)
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, subreddit_ids, id_type_ids, annotator_ids, labels=None, **kwargs):
        """Forward pass combining text and discrete features for classification."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
       
        text_embed = outputs.last_hidden_state[:, 0, :]

        # Discrete feature embedding
        sub_embed = self.subreddit_embedding(subreddit_ids.long())
        id_embed = self.id_type_embedding(id_type_ids.long())
        annotator_embed = self.annotator_embedding(annotator_ids.long())

        combined = torch.cat((text_embed, sub_embed, id_embed, annotator_embed), dim=1)
        combined = self.fusion_norm(combined)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

def load_data(file_path):
    """Loads raw JSONL data and fits LabelEncoders for metadata."""
    print(f"Loading data from {file_path}...")
    data = []

    feats = {"sub": [], "id": [], "anno": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get('conspiracy') not in ["Yes", "No"]: continue

                raw_id = item.get('id', 'unknown')
                id_prefix = raw_id.split('_')[0] if '_' in raw_id else 'unknown'
                sub = item.get('subreddit', 'unknown')
                anno = item.get('annotator', 'unknown')

                feats["sub"].append(sub)
                feats["id"].append(id_prefix)
                feats["anno"].append(anno)

                data.append({
                    "text": item['text'],
                    "label": 1 if item['conspiracy'] == "Yes" else 0,
                    "sub_raw": sub,
                    "id_raw": id_prefix,
                    "anno_raw": anno
                })
            except:
                pass

    print(f"Loaded {len(data)} valid samples.")

    encoders = {}
    for key in feats:
        uniques = list(set(feats[key]))
        if "unknown" not in uniques: uniques.append("unknown")
        le = LabelEncoder()
        le.fit(uniques)
        encoders[key] = le

    return data, encoders


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    raw_data, encoders = load_data(TRAIN_FILE)

    joblib.dump(encoders, os.path.join(OUTPUT_ROOT, "encoders.joblib"))

    hf_dataset = Dataset.from_list(raw_data)
    dataset_split = hf_dataset.train_test_split(test_size=TEST_SIZE, seed=42)

    # circuit training
    results_summary = []

    for model_name in MODEL_ZOO:
        print(f"\n\n{'=' * 40}\nProcessing: {model_name}\n{'=' * 40}")

        try:
            # --- A. Tokenizer ---
            if "deberta-v3" in model_name:
                tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)


            def preprocess(batch):
                encodings = tokenizer(batch['text'], truncation=True, padding="max_length", max_length=MAX_LEN)
                encodings['subreddit_ids'] = encoders['sub'].transform(batch['sub_raw'])
                encodings['id_type_ids'] = encoders['id'].transform(batch['id_raw'])
                encodings['annotator_ids'] = encoders['anno'].transform(batch['anno_raw'])
                encodings['labels'] = batch['label']
                return encodings


            # Process the dataset
            encoded_train = dataset_split['train'].map(preprocess, batched=True)
            encoded_eval = dataset_split['test'].map(preprocess, batched=True)

            cols = ['input_ids', 'attention_mask', 'labels', 'subreddit_ids', 'id_type_ids', 'annotator_ids']
            encoded_train.set_format(type='torch', columns=cols)
            encoded_eval.set_format(type='torch', columns=cols)

  
            print("Loading base model...")
            base_config = AutoConfig.from_pretrained(model_name)
            base_model_obj = AutoModel.from_pretrained(model_name)

            hidden_size = getattr(base_config, "hidden_size", 768)

            model = SimpleHybridModel(
                base_model=base_model_obj,
                num_labels=2,
                hidden_size=hidden_size,
                num_subreddits=len(encoders['sub'].classes_),
                num_id_types=len(encoders['id'].classes_),
                num_annotators=len(encoders['anno'].classes_)
            )

            # Trainer 
            safe_name = model_name.replace("/", "-")
            run_dir = os.path.join(OUTPUT_ROOT, safe_name)

            args = TrainingArguments(
                output_dir=run_dir,
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=50,
                learning_rate=LEARNING_RATE,
                fp16=torch.cuda.is_available(),  
                report_to="none",
                load_best_model_at_end=True,  
                metric_for_best_model="f1",
                save_total_limit=1,
                remove_unused_columns=False  
            )


            def compute_metrics(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                return evaluate.load("f1").compute(predictions=preds, references=p.label_ids, average="binary")


            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=encoded_train,
                eval_dataset=encoded_eval,
                compute_metrics=compute_metrics,
            )

            # Training
            trainer.train()

            metrics = trainer.evaluate()
            f1 = metrics['eval_f1']
            print(f"Training Done. F1: {f1:.4f}")

        
            save_path = os.path.join(OUTPUT_ROOT, "saved_models", safe_name)
            if not os.path.exists(save_path): os.makedirs(save_path)


            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)
            base_config.save_pretrained(save_path)

            results_summary.append({"Model": model_name, "F1": f1, "Status": "Success"})

            del model, base_model_obj, trainer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error on {model_name}: {e}")
            results_summary.append({"Model": model_name, "F1": 0.0, "Status": "Failed"})

    print("\nFinal Leaderboard:")
    for res in sorted(results_summary, key=lambda x: x['F1'], reverse=True):
        print(f"{res['Model']}: {res['F1']:.4f} ({res['Status']})")