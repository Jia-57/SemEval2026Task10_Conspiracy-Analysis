import argparse
import json
import torch
import random
import os
import re
import ast
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils1 import build_prompt, find_offsets

def load_jsonl(path):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_few_shot_examples(train_file_path):
    """Retrieves a balanced set of few-shot examples from the training data[cite: 102]."""
    data = load_jsonl(train_file_path)
    yes = [d for d in data if d.get('conspiracy') == 'Yes']
    no = [d for d in data if d.get('conspiracy') == 'No']
    examples = []
    if yes: examples.append(random.choice(yes))
    if no: examples.append(random.choice(no))
    while len(examples) < 3 and data:
        examples.append(random.choice(data))
    return examples

def parse_model_output(generated_text, original_text, _id):
    """Extracts and validates JSON content from the model output with robust fallback parsing[cite: 26]."""
    consp = 'No'
    markers = []
    try:
        content = generated_text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        s = content.find('{')
        e = content.rfind('}')
        
        if s == -1: return 'No', []

        # Robust handling for truncated JSON outputs
        if e == -1 or e < s:
            json_str = content[s:].strip()
            if not json_str.endswith(']}'):
                if not json_str.endswith(']'): json_str += ']}'
                else: json_str += '}'
        else:
            json_str = content[s:e+1]
        
        pred = None
        try:
            pred = json.loads(json_str)
        except:
            # Fallback: using regex to extract key classification and markers
            try:
                c_match = re.search(r'"conspiracy":\s*"(Yes|No)"', json_str, re.I)
                consp_val = c_match.group(1) if c_match else "No"
                m_texts = re.findall(r'"text":\s*"([^"]+)"', json_str)
                pred = {"conspiracy": consp_val, "markers": [{"text": t, "type": "Evidence"} for t in m_texts]}
            except:
                return 'No', []

        if not pred or not isinstance(pred, dict): return 'No', []

        raw_consp = str(pred.get('conspiracy', 'No')).strip().lower()
        consp = 'Yes' if 'yes' in raw_consp else 'No'
        
        raw_markers = pred.get('markers', [])
        if isinstance(raw_markers, list):
            markers = find_offsets(original_text, raw_markers)

    except Exception as e:
        print(f"[ID:{_id}] Parse error: {e}")
        return 'No', []
    return consp, markers

def main(args):
    """Orchestrates the model inference pipeline, from loading weights to saving prediction results[cite: 91]."""
    test_data = load_jsonl(args.test_binary_file)
    print(f"Loaded {len(test_data)} items.")

    few_shot_examples = None
    if args.stage == "B":
        few_shot_examples = get_few_shot_examples(args.train_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Increase maximum length to accommodate few-shot prompts
    tokenizer.model_max_length = 2048

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    if args.stage == "C" and args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    
    model.eval()

    bin_filename = f"submission_binary_{args.model_alias}_{args.stage}.jsonl"
    span_filename = f"submission_span_{args.model_alias}_{args.stage}.jsonl"

    with open(bin_filename, 'w', encoding='utf-8') as f_bin, \
         open(span_filename, 'w', encoding='utf-8') as f_span:

        for i, item in enumerate(tqdm(test_data)):
            _id, text = item.get('_id', 'unknown'), item.get('text', '')
            consp, markers = 'No', []
            try:
                messages = build_prompt(args.stage, text, few_shot_examples)
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs, 
                        max_new_tokens=1024, 
                        temperature=0.1, 
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                consp, markers = parse_model_output(generated, text, _id)
            except Exception as e:
                if "memory" in str(e): torch.cuda.empty_cache()
                consp, markers = 'No', []

            f_bin.write(json.dumps({"_id": _id, "conspiracy": consp}) + '\n')
            f_span.write(json.dumps({"_id": _id, "conspiracy": None, "markers": markers}) + '\n')
            if (i + 1) % 10 == 0:
                f_bin.flush(); f_span.flush()

    print(f"Done. Files: {bin_filename}, {span_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_alias", type=str, default="model")
    parser.add_argument("--stage", choices=["A1", "A2", "B", "C"], required=True)
    parser.add_argument("--test_binary_file", type=str, default="test_binary_rehydrated.jsonl")
    parser.add_argument("--test_span_file", type=str, default="test_span_rehydrated.jsonl")
    parser.add_argument("--train_file", type=str, default="train_cleaned_final.jsonl")
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter for Stage C")
    args = parser.parse_args()
    main(args)