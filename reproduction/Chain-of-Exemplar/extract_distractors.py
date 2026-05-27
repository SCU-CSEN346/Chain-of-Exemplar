"""
extract_distractors.py

Parses the raw pred and GT files into a clean aligned JSON file that all
downstream scoring scripts consume. Handles variable distractor counts
(1, 2, or 3) without penalizing shorter outputs.

Output format (extracted.json):
[
  {
    "id": "identity_test_0",
    "correct_answer": "apostrophe",
    "pred_distractors": ["antithesis", "pathos"],
    "gt_distractors": ["chiasmus"],
    "n_pred": 2,
    "n_gt": 1,
    "empty_pred": false
  },
  ...
]
"""

import json
import re
import argparse
from pathlib import Path


def parse_distractors(text: str) -> list[str]:
    """
    Parse numbered distractors from model output or GT assistant turn.
    Handles formats:
      (1) foo\n(2) bar\n(3) baz
      (1) foo (2) bar
    Returns a list of distractor strings. Empty list if none found.
    """
    if not text or not text.strip():
        return []

    # Primary: newline-separated numbered items
    matches = re.findall(
        r'\(\d+\)\s*(.+?)(?=\n\s*\(\d+\)|\Z)',
        text.strip(),
        re.DOTALL
    )
    result = [m.strip() for m in matches if m.strip()]

    # Fallback: inline format "(1) foo (2) bar" on one line
    if not result:
        matches = re.findall(r'\(\d+\)\s*([^(]+)', text)
        result = [m.strip() for m in matches if m.strip()]

    return result


def extract_correct_answer(user_prompt: str) -> str:
    """
    Extract the correct answer from the GT user conversation turn.
    The prompt contains a line like:
      Answer: apostrophe
    This appears before the exemplar section.
    """
    # Match the first "Answer:" occurrence (the actual answer, not the exemplar's)
    match = re.search(r'\nAnswer:\s*(.+?)(?=\n)', user_prompt)
    if match:
        return match.group(1).strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred',
        default='pred_test_dg_2ep_self_consistent_3samples_3distractors_final.json',
        help='Path to inference prediction JSON file'
    )
    parser.add_argument(
        '--gt',
        default='ScienceQA_test_dg_from_qg_rg_2ep_self_consistent_3samples_3distractors_final.json',
        help='Path to ground truth conversation JSON file'
    )
    parser.add_argument(
        '--output',
        default='extracted.json',
        help='Output path for extracted structured data'
    )
    args = parser.parse_args()

    with open(args.pred) as f:
        pred_data = json.load(f)
    with open(args.gt) as f:
        gt_data = json.load(f)

    # Build lookup from GT by id
    gt_lookup = {item['id']: item for item in gt_data}

    results = []
    mismatched_ids = 0
    missing_answer = 0

    for pred_item in pred_data:
        pid = pred_item['id']

        if pid not in gt_lookup:
            mismatched_ids += 1
            continue

        gt_item = gt_lookup[pid]
        user_prompt = gt_item['conversations'][0]['value']
        gt_assistant = gt_item['conversations'][1]['value']

        correct_answer = extract_correct_answer(user_prompt)
        if not correct_answer:
            missing_answer += 1

        pred_distractors = parse_distractors(pred_item['response'])
        gt_distractors = parse_distractors(gt_assistant)

        results.append({
            'id': pid,
            'correct_answer': correct_answer,
            'pred_distractors': pred_distractors,
            'gt_distractors': gt_distractors,
            'n_pred': len(pred_distractors),
            'n_gt': len(gt_distractors),
            'empty_pred': len(pred_distractors) == 0
        })

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Extracted {len(results)} samples -> {args.output}")
    print(f"  Mismatched IDs skipped: {mismatched_ids}")
    print(f"  Missing correct answer: {missing_answer}")
    print(f"  Empty predictions: {sum(1 for r in results if r['empty_pred'])}")

    # Distractor count summary
    from collections import Counter
    pred_counts = Counter(r['n_pred'] for r in results)
    print("\nPrediction distractor count distribution:")
    for k in sorted(pred_counts):
        print(f"  {k}: {pred_counts[k]} samples ({pred_counts[k]/len(results)*100:.1f}%)")


if __name__ == '__main__':
    main()