import json
import argparse
from tqdm import tqdm
from collections import Counter
import re
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import math

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data (json)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results (json)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of rationales to sample per input')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--chunk_size', type=int, default=2,
                        help='Number of samples to generate at once per example (for memory efficiency)')
    # Removed sharded_inference argument; only standard loading is supported
    return parser.parse_args()

# --- Answer Extraction ---
def extract_answer(rationale):
    match = re.search(r"Answer[:：]?\s*([A-D])", rationale)
    if match:
        return match.group(1)
    match = re.search(r"Answer[:：]?\s*(.*)", rationale)
    if match:
        return match.group(1).strip()
    return None

# --- Main Inference Function ---
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    results = []
    num_examples = len(data)
    batch_size = args.batch_size
    num_samples = args.num_samples
    chunk_size = args.chunk_size

    for batch_start in tqdm(range(0, num_examples, batch_size)):
        batch_examples = data[batch_start:batch_start+batch_size]
        batch_queries = []
        batch_ids = []
        for example in batch_examples:
            query = example['conversations'][0]['value'] if 'conversations' in example else example['question']
            # Patch: Replace <img> URLs with local file paths in data/images/ for all splits
            def replace_img_url_with_local(match):
                url = match.group(1)
                parts = url.split('/')
                if len(parts) >= 2:
                    subdir = parts[-2]
                    filename = parts[-1]
                    local_path = f"data/images/{subdir}/{filename}"
                else:
                    local_path = f"data/images/{parts[-1]}"
                return f"<img>{local_path}</img>"
            query = re.sub(r"<img>(.*?)</img>", replace_img_url_with_local, query)
            batch_queries.append(query)
            batch_ids.append(example.get('id', None))
        # Chunked generation for self-consistency
        batch_rationales = [[] for _ in range(len(batch_queries))]
        batch_answers = [[] for _ in range(len(batch_queries))]
        for sample_start in range(0, num_samples, chunk_size):
            this_chunk = min(chunk_size, num_samples - sample_start)
            expanded_queries = []
            expanded_id_refs = []
            for idx, q in enumerate(batch_queries):
                expanded_queries.extend([q]*this_chunk)
                expanded_id_refs.extend([idx]*this_chunk)
            encoded = tokenizer(expanded_queries, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            input_ids = encoded.input_ids.to(args.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id,
                )
            for i, output in enumerate(outputs):
                idx = expanded_id_refs[i]
                input_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
                rationale = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                batch_rationales[idx].append(rationale)
                answer = extract_answer(rationale)
                batch_answers[idx].append(answer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Aggregate results for this batch
        for i in range(len(batch_queries)):
            answer_counts = Counter([a for a in batch_answers[i] if a is not None])
            final_answer = answer_counts.most_common(1)[0][0] if answer_counts else None
            results.append({
                'id': batch_ids[i],
                'question': batch_queries[i],
                'final_answer': final_answer,
                'all_answers': batch_answers[i],
                'all_rationales': batch_rationales[i]
            })
        # Free up unused GPU memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
