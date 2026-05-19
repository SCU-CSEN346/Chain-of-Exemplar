"""
scoring_accuracy.py

Implements the paper's Accuracy metric:
  - For each test sample, replace the original ScienceQA wrong options
    with the generated distractors
  - Run a pretrained multimodal QA model on the modified MCQ
  - Accuracy = % of questions the QA model answers correctly
  - LOWER accuracy = HARDER distractors = BETTER generation quality

Variable distractor count handling (answering the key question):
  Samples with fewer than 3 predicted distractors are NOT penalized.
  Instead, the MCQ is built with however many distractors were generated
  (1, 2, or 3), plus the correct answer. The QA model still has to pick
  the right answer from a smaller option set — which actually makes it
  EASIER, so under-generation naturally shows up as higher accuracy
  (worse score) without any artificial penalty being needed.

  Samples with 0 predicted distractors are skipped by default
  (--skip_empty flag). Use --count_empty_as_correct to treat them as
  QA model getting the answer right (conservative: inflates accuracy,
  penalizes empty outputs).

Requires:
  - extracted.json from extract_distractors.py
  - problems.json from ScienceQA
  - A QA model checkpoint (Multimodal-CoT trained on ScienceQA)
    OR use --mock_mode to test the pipeline without a GPU

Usage:
  # Test pipeline without GPU:

  # Real run with QA model:
  python scoring_accuracy.py \
    --extracted extracted_distractors.json \
    --problems data/scienceqa/problems_blip2xl_angle.json \
    --qa_model WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_multitask_blip2xl_angle_2ep \
    --image_dir data/images \
    --output accuracy_results.json
"""

import json
import re
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
 
 
# ─────────────────────────────────────────────
# MCQ Construction
# ─────────────────────────────────────────────
 
CHOICE_LABELS = ['A', 'B', 'C', 'D']
 
 
def build_modified_mcq(correct_answer: str,
                       pred_distractors: list[str],
                       problem: dict,
                       qid: str,
                       seed: int = 42) -> dict | None:
    """
    Build a modified MCQ by replacing original wrong options with
    generated distractors. Correct answer is always included.
 
    If pred_distractors is empty, returns None (caller skips or handles).
 
    For 1 or 2 distractors: builds a 2- or 3-option MCQ.
    For 3 distractors: builds a 4-option MCQ (standard).
    For 4+ distractors: truncates to 3.
 
    Returns dict with:
      - choices: list of option strings (correct + distractors, shuffled)
      - correct_index: int index of correct answer in choices
      - correct_label: 'A'/'B'/'C'/'D'
      - question, context, image_path
    """
    if not pred_distractors:
        return None
 
    distractors = pred_distractors[:3]  # cap at 3
 
    # Combine correct answer with distractors
    all_options = [correct_answer] + distractors
 
    # Shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(all_options)
 
    correct_index = all_options.index(correct_answer)
    correct_label = CHOICE_LABELS[correct_index]
 
    return {
        'qid': qid,
        'question': problem.get('question', ''),
        'context': problem.get('hint', ''),
        'image': problem.get('image', None),
        'choices': all_options,
        'correct_index': correct_index,
        'correct_label': correct_label,
        'n_options': len(all_options)
    }
 
 
def format_qa_prompt(mcq: dict, image_dir: str) -> str:
    """
    Format the MCQ as a prompt for the Qwen-VL / Multimodal-CoT model.
    Matches the prompt format used in the CoE repo's training data.
    """
    choices_str = '\n'.join([
        f"({CHOICE_LABELS[i]}) {choice}"
        for i, choice in enumerate(mcq['choices'])
    ])
 
    context_part = f"Context: {mcq['context']}\n" if mcq['context'] else ""
 
    if mcq['image']:
        def replace_img_url_with_local(match):
            url = match.group(1)
            parts = url.split('/')
            if len(parts) >= 2:
                subdir = parts[-2]
                filename = parts[-1]
                return f"data/images/{subdir}/{filename}"  # return plain path, no tags
            else:
                return f"data/images/{parts[-1]}"

        # Strip any existing tags first, then sub on plain URL
        raw_image = mcq['image'].replace('<img>', '').replace('</img>', '').strip()
        local_path = replace_img_url_with_local(type('m', (), {'group': lambda self, i: raw_image})())

        prompt = (
            f"Picture: <img>{local_path}</img>\n"   # wrap once here
            f"{context_part}"
            f"Question: {mcq['question']}\n"
            f"Options:\n{choices_str}\n"
            f"Answer:"
        )
    # if mcq['image']:
    #     def replace_img_url_with_local(match):
    #         url = match.group(1)
    #         parts = url.split('/')
    #         if len(parts) >= 2:
    #             subdir = parts[-2]
    #             filename = parts[-1]
    #             local_path = f"data/images/{subdir}/{filename}"
    #         else:
    #             local_path = f"data/images/{parts[-1]}"
    #         return f"<img>{local_path}</img>"
    #     image_path = re.sub(r'<img>(.*?)</img>', replace_img_url_with_local, f"<img>{mcq['image']}</img>")
    #     prompt = (
    #         f"Picture: <img>{image_path}</img>\n"
    #         f"{context_part}"
    #         f"Question: {mcq['question']}\n"
    #         f"Options:\n{choices_str}\n"
    #         f"Answer:"
    #     )
    else:
        prompt = (
            f"{context_part}"
            f"Question: {mcq['question']}\n"
            f"Options:\n{choices_str}\n"
            f"Answer:"
        )
 
    return prompt
 
 
def parse_model_answer(response: str) -> str | None:
    """
    Extract a choice label (A/B/C/D) from the QA model's response.
    Tries multiple patterns in order of specificity.
    Returns None if parsing fails.
    """
    if not response:
        return None
 
    # Pattern 1: explicit label in parens "(A)" or "(B)"
    match = re.search(r'\(([ABCD])\)', response)
    if match:
        return match.group(1)
 
    # Pattern 2: "The answer is A" or "Answer: B"
    match = re.search(r'(?:answer is|Answer:)\s*([ABCD])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
 
    # Pattern 3: bare label at start of response
    stripped = response.strip().upper()
    if stripped and stripped[0] in CHOICE_LABELS:
        return stripped[0]
 
    return None
 
 
# ─────────────────────────────────────────────
# Mock QA Model (for pipeline testing without GPU)
# ─────────────────────────────────────────────
 
class MockQAModel:
    """
    Simulates a QA model that randomly picks an answer.
    Used for testing the pipeline end-to-end without a real model.
    Expected accuracy: ~25% for 4-option, ~33% for 3-option, ~50% for 2-option.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
 
    def predict(self, prompt: str, n_options: int) -> str:
        label = self.rng.choice(CHOICE_LABELS[:n_options])
        return f"({label})"
 
 
# ─────────────────────────────────────────────
# Real QA Model Loader
# ─────────────────────────────────────────────
 
def load_qa_model(model_path: str):
    """
    Load the Multimodal-CoT QA model (Qwen-VL based).
    Returns (model, tokenizer).
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
 
    print(f"Loading QA model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    print("QA model loaded.")
    return model, tokenizer
 
 
def real_predict(model, tokenizer, prompt: str) -> str:
    """Run real inference with the QA model."""
    response, _ = model.chat(tokenizer, query=prompt, history=None)
    return response
 
 
# ─────────────────────────────────────────────
# Slicing helpers
# ─────────────────────────────────────────────
 
def get_slices(problem: dict) -> dict:
    subject = problem.get('subject', 'unknown')
    grade_str = problem.get('grade', 'grade1')
    grade_num = int(re.search(r'\d+', grade_str).group()) if re.search(r'\d+', grade_str) else 1
    has_image = problem.get('image') is not None
    has_hint = bool(problem.get('hint', '').strip())
 
    if has_image and not has_hint:
        modality = 'img'
    elif has_hint and not has_image:
        modality = 'txt'
    elif has_image and has_hint:
        modality = 'txt'
    else:
        modality = 'no'
 
    return {
        'subject': subject,
        'grade': 'g16' if grade_num <= 6 else 'g712',
        'modality': modality
    }
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted', required=True,
                        help='Path to extracted.json from extract_distractors.py')
    parser.add_argument('--problems', default=None,
                        help='Path to ScienceQA problems.json. Required for real mode, '
                             'optional in --mock_mode (skips sliced results)')
    parser.add_argument('--image_dir', default='data/scienceqa',
                        help='Root directory of ScienceQA images')
    parser.add_argument('--qa_model', default=None,
                        help='HuggingFace model ID or local path to QA model checkpoint')
    parser.add_argument('--mock_mode', action='store_true',
                        help='Use a random mock QA model (no GPU needed, for testing)')
    parser.add_argument('--output', default='accuracy_results.json',
                        help='Output path for detailed per-sample results')
    parser.add_argument('--skip_empty', action='store_true', default=True,
                        help='Skip samples with 0 predicted distractors (default: True)')
    parser.add_argument('--count_empty_as_correct', action='store_true',
                        help='Count empty predictions as QA model getting it right '
                             '(penalizes empty outputs by inflating accuracy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for option shuffling')
    args = parser.parse_args()
 
    # Load data
    with open(args.extracted) as f:
        data = json.load(f)
 
    # problems.json is optional in mock mode
    problems = {}
    if args.problems:
        with open(args.problems) as f:
            problems = json.load(f)
    elif not args.mock_mode:
        raise ValueError("--problems is required when not using --mock_mode")
 
    # Load QA model
    if args.mock_mode:
        print("Running in MOCK MODE — results are random baselines, not real accuracy.")
        mock_model = MockQAModel(seed=args.seed)
        qa_model, qa_tokenizer = None, None
    elif args.qa_model:
        qa_model, qa_tokenizer = load_qa_model(args.qa_model)
        mock_model = None
    else:
        raise ValueError("Must provide either --qa_model or --mock_mode")
 
    # Accumulators
    results = []
    slices = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall = {'correct': 0, 'total': 0, 'skipped': 0, 'parse_failures': 0}
 
    # Pre-compute a robust QID lookup that tries multiple key formats.
    # problems.json keys vary across repo versions:
    #   "0", "1", ...            (plain integer strings)
    #   0, 1, ...                (integer keys — after json.load these are strings)
    #   "test_0", "test_1", ...  (split-prefixed)
    #   "identity_test_0", ...   (full id)
    def find_problem(raw_id: str) -> dict | None:
        if not problems:
            return None
        stripped = raw_id.replace('identity_test_', '')  # e.g. "0"
        for candidate in [
            stripped,                          # "0"
            raw_id,                            # "identity_test_0"
            f"test_{stripped}",                # "test_0"
            f"train_{stripped}",               # "train_0"
            str(int(stripped)) if stripped.isdigit() else None,  # normalise "00" -> "0"
        ]:
            if candidate and candidate in problems:
                return problems[candidate]
        return None
 
    # Warn early if zero lookups succeed (catches key format mismatch before full run)
    n_probe = min(20, len(data))
    n_found = sum(1 for item in data[:n_probe] if find_problem(item['id']) is not None)
    if n_found == 0 and problems:
        sample_keys = list(problems.keys())[:5]
        sample_ids  = [item['id'] for item in data[:5]]
        print(f"WARNING: problems.json key format does not match extracted IDs.")
        print(f"  problems.json sample keys : {sample_keys}")
        print(f"  extracted.json sample ids : {sample_ids}")
        print(f"  Stripped sample qids      : {[i.replace('identity_test_','') for i in sample_ids]}")
        print(f"  All samples will be skipped. Fix the key format in find_problem() above.")
 
    for item in tqdm(data, desc="Scoring"):
        pred_d = item['pred_distractors']
        correct_answer = item['correct_answer']
 
        # Handle empty predictions
        if not pred_d:
            overall['skipped'] += 1
            if args.count_empty_as_correct:
                # Penalize: count as QA model getting it right
                overall['correct'] += 1
                overall['total'] += 1
                results.append({
                    'id': item['id'],
                    'skipped': True,
                    'reason': 'empty_prediction',
                    'qa_correct': True
                })
            continue
 
        # Look up problem metadata
        problem = find_problem(item['id'])
        if problem is None:
            if args.mock_mode:
                # Stub so mock mode always scores every non-empty sample
                problem = {
                    'question': correct_answer,
                    'hint': '',
                    'image': None,
                    'subject': 'natural science',
                    'grade': 'grade1'
                }
            else:
                overall['skipped'] += 1
                continue
 
        qid = item['id'].replace('identity_test_', '')
 
        # Build modified MCQ
        mcq = build_modified_mcq(
            correct_answer=correct_answer,
            pred_distractors=pred_d,
            problem=problem,
            qid=qid,
            seed=args.seed
        )
 
        if mcq is None:
            overall['skipped'] += 1
            continue
 
        # Format prompt and run QA model
        prompt = format_qa_prompt(mcq, args.image_dir)
 
        if args.mock_mode:
            response = mock_model.predict(prompt, mcq['n_options'])
        else:
            response = real_predict(qa_model, qa_tokenizer, prompt)
 
        # Parse model answer
        predicted_label = parse_model_answer(response)
 
        if predicted_label is None:
            overall['parse_failures'] += 1
            overall['skipped'] += 1
            continue
 
        # Check correctness
        is_correct = (predicted_label == mcq['correct_label'])
 
        overall['correct'] += is_correct
        overall['total'] += 1
 
        # Slice tracking
        sk = get_slices(problem)
        for slice_type, slice_val in sk.items():
            key = f"{slice_type}:{slice_val}"
            slices[key]['correct'] += is_correct
            slices[key]['total'] += 1
 
        results.append({
            'id': item['id'],
            'qid': qid,
            'skipped': False,
            'n_pred_distractors': item['n_pred'],
            'correct_answer': correct_answer,
            'pred_distractors': pred_d,
            'choices': mcq['choices'],
            'correct_label': mcq['correct_label'],
            'predicted_label': predicted_label,
            'qa_correct': is_correct,
            'raw_response': response if args.mock_mode else response[:200]
        })
 
    # ─── Print Results ───
    n = overall['total']
    c = overall['correct']
    acc = c / n * 100 if n > 0 else 0
 
    print(f"\n{'='*55}")
    print(f"ACCURACY RESULTS ({'MOCK' if args.mock_mode else 'REAL'})")
    print(f"{'='*55}")
    print(f"  Overall Accuracy:  {acc:.2f}%  ({c}/{n})")
    print(f"  Skipped (empty/missing): {overall['skipped']}")
    print(f"  Parse failures:    {overall['parse_failures']}")
    print(f"\n  NOTE: Lower accuracy = harder distractors = better generation")
 
    print(f"\n  Sliced results:")
    for key in sorted(slices.keys()):
        s = slices[key]
        if s['total'] > 0:
            slice_acc = s['correct'] / s['total'] * 100
            print(f"    {key}: {slice_acc:.2f}%  (n={s['total']})")
 
    # Save detailed results
    output = {
        'overall_accuracy': acc,
        'n_scored': n,
        'n_correct': c,
        'n_skipped': overall['skipped'],
        'n_parse_failures': overall['parse_failures'],
        'mock_mode': args.mock_mode,
        'slices': {
            k: {
                'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0,
                'n': v['total']
            }
            for k, v in slices.items()
        },
        'samples': results
    }
 
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
 
    print(f"\nDetailed results saved to: {args.output}")
 
 
if __name__ == '__main__':
    main()
 