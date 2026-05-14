import json
import os

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


DATA_ROOT = "data/scienceqa"
RETRIEVE_DIR = "retrieve"

CONTEXT_EMB_PATH = os.path.join(RETRIEVE_DIR, "context_embedding.json")
ANSWER_EMB_PATH = os.path.join(RETRIEVE_DIR, "answer_embedding.json")
QUESTION_EMB_PATH = os.path.join(RETRIEVE_DIR, "question_embedding.json")


model = SentenceTransformer("all-mpnet-base-v2", device="cuda")


def save_embedding(problems):
    os.makedirs(RETRIEVE_DIR, exist_ok=True)

    context_dict = {}
    answer_dict = {}
    question_dict = {}

    train_items = {k: v for k, v in problems.items() if v["split"] == "train"}

    for qid, item in tqdm(train_items.items(), desc="Encoding train embeddings"):
        context = item["hint"] if item["hint"] != "" else ""
        answer = item["choices"][item["answer"]]
        question = item["question"]

        context_dict[qid] = model.encode(context, convert_to_tensor=False).tolist()
        answer_dict[qid] = model.encode(answer, convert_to_tensor=False).tolist()
        question_dict[qid] = model.encode(question, convert_to_tensor=False).tolist()

    with open(CONTEXT_EMB_PATH, "w") as fp:
        json.dump(context_dict, fp)

    with open(ANSWER_EMB_PATH, "w") as fp:
        json.dump(answer_dict, fp)

    with open(QUESTION_EMB_PATH, "w") as fp:
        json.dump(question_dict, fp)


def retrieve_example():
    problems_path = os.path.join(DATA_ROOT, "problems.json")
    output_path = os.path.join(DATA_ROOT, "problems_blip2xl_angle.json")

    problems = json.load(open(problems_path))

    if not os.path.exists(ANSWER_EMB_PATH) or not os.path.exists(QUESTION_EMB_PATH):
        save_embedding(problems)

    answer_embedding = json.load(open(ANSWER_EMB_PATH))
    question_embedding = json.load(open(QUESTION_EMB_PATH))

    candidate_ids = list(answer_embedding.keys())

    answer_matrix = torch.tensor(
        [answer_embedding[qid] for qid in candidate_ids],
        dtype=torch.float32,
        device="cuda",
    )

    question_matrix = torch.tensor(
        [question_embedding[qid] for qid in candidate_ids],
        dtype=torch.float32,
        device="cuda",
    )

    answer_matrix = torch.nn.functional.normalize(answer_matrix, dim=1)
    question_matrix = torch.nn.functional.normalize(question_matrix, dim=1)

    k = 5
    updated = 0

    for qid, item in tqdm(problems.items(), desc="Retrieving exemplars"):
        if item["split"] == "train" and item["hint"] == "" and item["image"] is None:
            answer_vec = torch.tensor(
                answer_embedding[qid],
                dtype=torch.float32,
                device="cuda",
            ).unsqueeze(0)

            question_vec = torch.tensor(
                question_embedding[qid],
                dtype=torch.float32,
                device="cuda",
            ).unsqueeze(0)

            answer_vec = torch.nn.functional.normalize(answer_vec, dim=1)
            question_vec = torch.nn.functional.normalize(question_vec, dim=1)

            scores = (answer_vec @ answer_matrix.T).squeeze(0) + (
                question_vec @ question_matrix.T
            ).squeeze(0)

            self_idx = candidate_ids.index(qid)
            scores[self_idx] = -999.0

            top_indices = torch.topk(scores, k=k).indices.tolist()
            top_ids = [candidate_ids[i] for i in top_indices]

            item["relevant_question"] = top_ids
            updated += 1
        else:
            item["relevant_question"] = []

    with open(output_path, "w") as fp:
        json.dump(problems, fp)

    print("saved:", output_path)
    print("updated train samples:", updated)


if __name__ == "__main__":
    retrieve_example()
