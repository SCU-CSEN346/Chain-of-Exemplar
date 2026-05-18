import json
import argparse
import os
import re
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--min_new_tokens', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.4)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    return parser.parse_args()


def extract_answer_from_prompt(prompt):
    """Extract the answer from the 'Answer: X' line in conversations[0]."""
    match = re.search(r"^Answer:\s*(.+)", prompt, re.MULTILINE)
    if match:
        return match.group(1).split('\n')[0].strip()
    return None


def score_rationale(model, tokenizer, question, rationale, answer, device):
    prompt = f"{question}\n{rationale}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)
    answer_ids = tokenizer(
        str(answer),
        return_tensors="pt",
        max_length=64,
        truncation=True
    ).input_ids.to(device)
    if answer_ids[0, 0] == tokenizer.bos_token_id:
        answer_ids = answer_ids[:, 1:]
    input_ids = inputs.input_ids
    labels = torch.cat([torch.full_like(input_ids, -100), answer_ids], dim=1)
    full_input = torch.cat([input_ids, answer_ids], dim=1)
    with torch.no_grad():
        outputs = model(full_input, labels=labels)
    return -outputs.loss.item()


def apply_qwen_chat_template(prompt: str) -> str:
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    system_msg = "You are a helpful assistant. Directly answer the user's question with a concise reasoning process that leads to the final answer. Keep your response in English."
    return (
        f"{im_start}system\n{system_msg}{im_end}\n"
        f"{im_start}user\n{prompt}{im_end}\n"
        f"{im_start}assistant\n"
    )


def main():
    args = parse_args()

    # Safe makedirs: only call if there's actually a directory component
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()

    data = json.load(open(args.data_path))

    # Shard data for multi-GPU SLURM array jobs
    if args.num_shards > 1:
        per_shard = len(data) // args.num_shards
        start = args.shard_id * per_shard
        end = len(data) if args.shard_id == args.num_shards - 1 else start + per_shard
        data = data[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing examples {start}–{end} ({len(data)} total)", flush=True)

    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    results = []

    for example in tqdm(data):
        prompt = example["conversations"][0]["value"]

        # Fix image paths to point at local files
        def replace_img_url_with_local(match):
            url = match.group(1)
            parts = url.split('/')
            if len(parts) >= 2:
                local_path = f"data/images/{parts[-2]}/{parts[-1]}"
            else:
                local_path = f"data/images/{parts[-1]}"
            return f"<img>{local_path}</img>"

        prompt = re.sub(r"<img>(.*?)</img>", replace_img_url_with_local, prompt)

        # Bug fix: extract the short answer from conversations[0], not conversations[1].
        # conversations[1] is the full gold rationale — wrong target for cross-entropy scoring.
        answer = extract_answer_from_prompt(prompt)
        if answer is None:
            print(f"WARNING: No answer found in prompt for id {example.get('id')}. Skipping.", flush=True)
            continue

        formatted_prompt = apply_qwen_chat_template(prompt)

        encoded = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(args.device)
        input_len = encoded.input_ids.shape[-1]

        outputs = model.generate(
            **encoded,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_return_sequences=args.num_return_sequences,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id,
        )

        all_rationales = [
            tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                     .replace("<|im_end|>", "")
                     .strip()
            for seq in outputs
        ]

        filtered = [r for r in all_rationales if r]
        if not filtered:
            best_rationale = "[NO_RATIONALE_GENERATED]"
            scores = [float('-inf')] * len(all_rationales)
        else:
            scores = [score_rationale(model, tokenizer, prompt, r, answer, args.device) for r in filtered]
            best_rationale = filtered[scores.index(max(scores))]

        result = {
            'id': example.get('id', None),
            'question': prompt,
            'answer': answer,
            'best_rationale': best_rationale,
            'all_rationales': all_rationales,
            'scores': scores,
        }
        results.append(result)

        # flush=True ensures this appears immediately in SLURM logs
        print(f"Result: {result}", flush=True)

    json.dump(results, open(args.output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved {args.output_path} ({len(results)} samples)", flush=True)
    print(results[0], flush=True)


if __name__ == '__main__':
    main()