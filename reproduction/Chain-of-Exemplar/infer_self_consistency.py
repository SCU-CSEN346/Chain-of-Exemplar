import json
import argparse
import os
from tqdm import tqdm
import re
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


def apply_qwen_chat_template(prompt: str, tokenizer) -> str:
    # Replicates QWen's internal chat formatting from model.chat()
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    system_msg = "You are a helpful assistant. Directly answer the user's question with a concise reasoning process that leads to the final answer. Keep your response in English."

    formatted = (
        f"{im_start}system\n{system_msg}{im_end}\n"
        f"{im_start}user\n{prompt}{im_end}\n"
        f"{im_start}assistant\n"
    )
    return formatted


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
    alt_answer_keys = ['answer', 'gt_answer', 'label', 'target', 'correct_answer']
    results = []

    for example in tqdm(data):
        prompt = example["conversations"][0]["value"]
        
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

        prompt = re.sub(r"<img>(.*?)</img>", replace_img_url_with_local, prompt)
        answer = extract_answer_from_prompt(prompt)
        if answer is None:
            print(f"WARNING: No answer found in prompt for id {example.get('id')}. Skipping.", flush=True)
            continue


        # Use tokenizer's chat template exactly like model.chat() does,
        # so the model sees the same format it was fine-tuned on
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = apply_qwen_chat_template(prompt, tokenizer)

        encoded = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        ).to(args.device)
        input_len = encoded.input_ids.shape[-1]

        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)

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

        # answer = None
        # for k in alt_answer_keys:
        #     if k in example:
        #         answer = example[k]
        #         break
        # if answer is None:
        #     raise ValueError(f"No answer field found in example id {example.get('id', None)}")

        filtered = [r for r in all_rationales if r]
        if not filtered:
            best_rationale = "[NO_RATIONALE_GENERATED]"
            scores = [float('-inf')] * len(all_rationales)
        else:
            scores = [score_rationale(model, tokenizer, prompt, r, answer, args.device) for r in filtered]
            best_rationale = filtered[scores.index(max(scores))]

        results.append({
            'id': example.get('id', None),
            'question': prompt,       # store the raw prompt, not the formatted one
            'best_rationale': best_rationale,
            'all_rationales': all_rationales,
            'scores': scores,
        })

        print(f"Result: {results[-1]}", flush=True)
        #print('Result: {}'.format(results[-1]))  # Print each result as it's generated for real-time monitoring

    json.dump(results, open(args.output_path, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved {args.output_path} ({len(results)} samples)", flush=True)
    #print(results[0], flush=True)

if __name__ == '__main__':
    main()

# import json
# import argparse
# from tqdm import tqdm
# import re
# from peft import AutoPeftModelForCausalLM
# from transformers import AutoTokenizer
# import torch
# import torch.nn.functional as F

# # --- Argument Parsing ---
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, required=True, help='Path to model directory')
#     parser.add_argument('--data_path', type=str, required=True, help='Path to input data (json)')
#     parser.add_argument('--output_path', type=str, required=True, help='Path to save results (jsonl)')
#     parser.add_argument('--num_return_sequences', type=int, default=3, help='Number of rationales to generate per input')
#     parser.add_argument('--max_new_tokens', type=int, default=256)
#     parser.add_argument('--min_new_tokens', type=int, default=10)
#     parser.add_argument('--top_p', type=float, default=0.95)
#     parser.add_argument('--top_k', type=int, default=20)
#     parser.add_argument('--repetition_penalty', type=float, default=1.2)
#     parser.add_argument('--no_repeat_ngram_size', type=int, default=4)
#     parser.add_argument('--temperature', type=float, default=0.7)
#     parser.add_argument('--device', type=str, default='cuda')
#     return parser.parse_args()


# # --- Main Inference Function ---
# def main():
#     args = parse_args()
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float32)
#     model = AutoPeftModelForCausalLM.from_pretrained(
#         args.model_path,
#         device_map="cuda" if torch.cuda.is_available() else "cpu",
#         trust_remote_code=True
#     ).eval()

#     with open(args.data_path, 'r') as f:
#         data = json.load(f)

#     # Helper: Cross-entropy scoring for rationales
#     def score_rationale(model, tokenizer, question, rationale, answer, device):
#         prompt = f"{question}\n{rationale}\nAnswer:"
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         answer_ids = tokenizer(str(answer), return_tensors="pt").input_ids.to(device)
#         if answer_ids[0, 0] == tokenizer.bos_token_id:
#             answer_ids = answer_ids[:, 1:]
#         input_ids = inputs.input_ids
#         labels = torch.cat([torch.full_like(input_ids, -100), answer_ids], dim=1)
#         full_input = torch.cat([input_ids, answer_ids], dim=1)
#         with torch.no_grad():
#             outputs = model(full_input, labels=labels)
#         return -outputs.loss.item()

#     import re

#     def clean_rationale(text):
#         # Remove runs of special characters with no alphanumeric content nearby
#         text = re.sub(r'([^\w\s])\1{2,}', '', text)        # remove repeated special chars
#         text = re.sub(r'\n{3,}', '\n\n', text)              # collapse excessive newlines
#         text = re.sub(r'[(),.\n]{6,}', ' ', text)           # remove long runs of punctuation
#         text = re.sub(r'\s{2,}', ' ', text)                 # collapse whitespace
#         return text.strip()

#     results = []
#     alt_answer_keys = ['answer', 'gt_answer', 'label', 'target', 'correct_answer']
#     bad_chars = ['\n', '(', ')', ',,', '..',  '\n\n', '-->', '()', '.jpg', '.png', '.jpeg', '.bmp', '.gif', '<', '<<', '>', '>>']
#     bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_chars if tokenizer.encode(w, add_special_tokens=False)]

#     def extract_answer_from_prompt(prompt):
#         match = re.search(r"Answer:\s*(.+)", prompt)
#         if match:
#             return match.group(1).split('\n')[0].strip()
#         return None

#     for example in tqdm(data):
#         query = "\nAnswer the question after 'Question:' directly and concisely show your thought process. Use the example as a guide for reasoning and not for format. Only pull content that directly answers the question from the example. Format your response in sentences only, don't use special characters apart from punctuation."
#         query += "\n" + example['conversations'][0]['value'] if 'conversations' in example else example['question']

#         def replace_img_url_with_local(match):
#             url = match.group(1)
#             parts = url.split('/')
#             if len(parts) >= 2:
#                 subdir = parts[-2]
#                 filename = parts[-1]
#                 local_path = f"data/images/{subdir}/{filename}"
#             else:
#                 local_path = f"data/images/{parts[-1]}"
#             return f"[IMAGE]{local_path}[/IMAGE]"

#         query = re.sub(r"<img>(.*?)</img>", replace_img_url_with_local, query)
#         print(query)

#         # Generate all sequences in a single call using num_return_sequences
#         encoded = tokenizer(query, return_tensors='pt', max_length=1024, truncation=True)
#         input_ids = encoded.input_ids.to(args.device)
#         input_len = input_ids.shape[-1]

#         outputs = model.generate(
#             input_ids=input_ids,
#             do_sample=True,
#             top_p=args.top_p,
#             top_k=args.top_k,
#             temperature=args.temperature,
#             max_new_tokens=args.max_new_tokens,
#             min_new_tokens=args.min_new_tokens,
#             repetition_penalty=args.repetition_penalty,
#             no_repeat_ngram_size=args.no_repeat_ngram_size,
#             num_return_sequences=args.num_return_sequences,
#             bad_words_ids=bad_words_ids,
#             pad_token_id=tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

#         # Decode each returned sequence, slicing off the input prompt tokens
#         all_rationales = [
#             clean_rationale(tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip())
#             #tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
#             for seq in outputs
#         ]

#         # Filter blanks and fall back to placeholder if all failed
#         filtered = [r for r in all_rationales if r and r != "[NO_RATIONALE_GENERATED]"]
#         if not filtered:
#             scores = [float('-inf')] * len(all_rationales)
#             best_rationale = "[NO_RATIONALE_GENERATED]"
#         else:
#             # Set answer from field or regex extraction
#             answer = None
#             for k in alt_answer_keys:
#                 if k in example:
#                     answer = example[k]
#                     break
#             if answer is None:
#                 answer = extract_answer_from_prompt(query)
#             if answer is None:
#                 raise ValueError(f"No answer field found or extracted in example id {example.get('id', None)}. Please check your input data format.")

#             scores = [score_rationale(model, tokenizer, query, r, answer, args.device) for r in filtered]
#             best_idx = scores.index(max(scores))
#             best_rationale = filtered[best_idx]

#         results.append({
#             'id': example.get('id', None),
#             'question': query,
#             'best_rationale': best_rationale,
#             'all_rationales': all_rationales,
#             'scores': scores,
#         })

#     with open(args.output_path, 'w') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

# if __name__ == '__main__':
#     main()