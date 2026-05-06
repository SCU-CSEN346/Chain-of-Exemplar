# using mplug environment

import pdb
import json
import jsonlines
import nltk
import os, string, re, json, argparse, evaluate
# from sentence_transformers import SentenceTransformer, util
# from angle_emb import AnglE, Prompts
# from bleu.bleu import Bleu
import pandas as pd
from tqdm import tqdm
from evaluate.utils.file_utils import DownloadConfig
from evaluate import load
# from metrics.bleu import bleu as Bleu
# from metrics.rouge import rouge as Rouge
# from metrics.meteor import meteor as Meteor
# from metrics.bleurt import bleurt as Bleurt
# import datasets
# nltk.download('wordnet')
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7,8,9'
def semantic_similarity(vec, vecs):
    vec = angle.encode({'text': vec}, to_numpy=True)
    vec = torch.tensor(vec[0])
    vecs = angle.encode({'text': vecs}, to_numpy=True)
    vecs = torch.tensor(vecs[0])
    return util.cos_sim(vec, vecs)[0][0].item()

def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def grade_score(df, bleurt):
    nls = []
    for curit, (q, gq) in enumerate(zip(df['question'], df['generated_question'])):
        result = bleurt.compute(predictions=[normalize(gq)], references=[normalize(q)])
        nls.append(result)
    return nls


def get_batch(iterable, n=1):
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ceildiv(a, b):
    # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python/17511341#17511341
    return -(a // -b)


def grade_score_with_batching(df, bleurt, batch_size=64, normalize_flag=True):
    # Add batching to speed up BLEURT model computation
    # Note: BLEURT metric is non commutative, therefore predictions must match questions generated
    # df['target'] = df['target'].apply(normalize)
    # if normalize_flag:
    #     df['generated_question'] = df['generated_question'].apply(normalize)

    # ref_q = df['target'].tolist()
    # gen_q = df['generated_question'].tolist()
    ref_q = df['target']
    gen_q = df['generated_question']

    scores = []
    num_batches = ceildiv(len(ref_q), batch_size)
    for ref_q_batch, gen_q_batch in tqdm( zip(get_batch(ref_q, batch_size), get_batch(gen_q, batch_size)), total=num_batches ):
        batch_scores = bleurt.compute(predictions=gen_q_batch, references=ref_q_batch)
        scores.extend(batch_scores["scores"])

    return scores
def ml_metrics(results):
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleurt = evaluate.load('bleurt', 'bleurt-20')

    bleu4s, meteors, rouges = [], [], []

    bleurt_scores = grade_score_with_batching(results, bleurt, 64)

    for _, ans in tqdm(results.iterrows(), total=results.shape[0]):
        ref = ans['target']
        hyp = ans['generated_question']
        bleu4s.append(bleu.compute(predictions=[hyp], references=[ref])['bleu'])
        meteors.append(meteor.compute(predictions=[hyp], references=[ref])['meteor'])
        rouges.append(rouge.compute(predictions=[hyp], references=[ref])['rougeL'])

    res_len = len(bleu4s)
    # b1, b2, b3, b4 = sum(bleu1s) / res_len, sum(bleu2s) / res_len, sum(bleu3s) / res_len, sum(bleu4s) / res_len
    b4 = sum(bleu4s) / res_len
    meteor_score = sum(meteors) / res_len
    rouge_l = sum(rouges) / res_len
    bleurt_score = sum(bleurt_scores) / res_len

    # print("BLEU-N-grams: 1-{:.4f}, 2-{:.4f}, 3-{:.4f}, 4-{:.4f}".format(b1, b2, b3, b4))
    print("BLEU-4: {:.4f}".format(b4))
    print("METEOR: {:.4f}".format(meteor_score))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("BLEURT: {:.4f}".format(bleurt_score))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--problems", required=True)
    args = parser.parse_args()

    gt = json.load(open(args.gt))
    generated_question = json.load(open(args.pred))
    problems = json.load(open(args.problems))

    target = [example['conversations'][1]['value'] for example in gt]
    assert len(target) == len(generated_question), f'target len: {len(target)} != predict len: {len(generated_question)}!!'

    split_qids = [qid for qid in problems if problems[qid]['split'] == 'test']

    def resolve_qid(i):
        raw_id = gt[i]['id']
        if raw_id.startswith('identity_'):
            return raw_id[len('identity_'):]
        if raw_id.startswith('qg_test_') or raw_id.startswith('rg_test_') or raw_id.startswith('dg_test_'):
            return split_qids[i]
        return raw_id

    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleurt = evaluate.load('bleurt', 'bleurt-20')

    bleu4s, meteors, rouges, refs, hyps = [], [], [], [], []

    for i, example in tqdm(enumerate(gt), total=len(gt)):
        qid = resolve_qid(i)
        meta = problems[qid]
        hyp = generated_question[i]['response']
        ref = target[i]

        if hyp == '':
            hyp = 'None'
        if ref == '':
            ref = 'None'

        bl = bleu.compute(predictions=[hyp], references=[ref])['bleu']
        me = meteor.compute(predictions=[hyp], references=[ref])['meteor']
        ro = rouge.compute(predictions=[hyp], references=[ref])['rougeL']

        bleu4s.append(bl)
        meteors.append(me)
        rouges.append(ro)
        refs.append(normalize(ref))
        hyps.append(normalize(hyp))

    results = {'target': refs, 'generated_question': hyps}
    bleurt_scores = grade_score_with_batching(results, bleurt, 64)

    res_len = len(bleurt_scores)
    b4 = sum(bleu4s) / res_len
    meteor_score = sum(meteors) / res_len
    rouge_l = sum(rouges) / res_len
    bleurt_score = sum(bleurt_scores) / res_len

    print('Total:')
    print("BLEU-4: {:.4f}".format(b4))
    print("METEOR: {:.4f}".format(meteor_score))
    print("ROUGE-L: {:.4f}".format(rouge_l))
    print("BLEURT: {:.4f}".format(bleurt_score))
