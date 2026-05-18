import json
import os
import argparse


def build_example(problem, id, data_root):
    answer = 'Answer: ' + problem['choices'][problem['answer']] + '\n'
    question = problem['question']
    rationale = problem['lecture'] + problem['solution']
    choices = problem['choices'].copy()
    choices.remove(problem['choices'][problem['answer']])
    distractors = ''
    for i, choice in enumerate(choices):
        distractors += f'({i + 1}) {choice}\n'
    if problem['image'] != None:
        image = 'Picture: <img>' + \
                os.path.join(data_root, problem['split'], str(id), problem['image']) + f'</img>\n'
        context = f'Context: ' + problem['hint'] + '\n' if problem['hint'] != '' else ''
        qg = '\nExample:\n' + image + context + answer + f'Question: {question}'
        rg = '\nExample:\n' + image + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = '\nExample:\n' + image + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
    else:
        context = 'Context: ' + problem['hint'] + '\n' if problem['hint'] != '' else ''
        qg = '\nExample:\n' + context + answer + f'Question: {question}'
        rg = '\nExample:\n' + context + f'Question: {question}\n' + answer + f'Reasoning: {rationale}'
        dg = '\nExample:\n' + context + f'Question: {question}\n' + f'Reasoning: {rationale}\n' + answer + f'Distractors: {distractors}'
    return qg, rg, dg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', type=str, default='infer/pred_test_qg_2ep.json', help='Questions JSON')
    parser.add_argument('--rationales', type=str, default='infer/rationales_self_consistent_3samples_final.json', help='Rationales JSON')
    parser.add_argument('--problems', type=str, default='data/scienceqa/problems_blip2xl_angle.json', help='Problems JSON')
    parser.add_argument('--output', type=str, default='data/ScienceQA_test_dg_from_qg_rg_2ep.json', help='Output JSON')
    parser.add_argument('--split', type=str, default='test', help='Split to use')
    args = parser.parse_args()

    with open(args.questions, 'r') as f:
        output_questions = json.load(f)
    with open(args.rationales, 'r') as f:
        output_rationales = json.load(f)
    with open(args.problems, 'r') as f:
        problems = json.load(f)

    data_root = os.path.dirname(args.problems) if os.path.dirname(args.problems) else 'data/scienceqa/'
    save_list = []
    id = 0

    for qid in problems:
        if problems[qid]['split'] == args.split:
            example_dict = {"id": f"identity_{qid}"}
            output_question = 'Question: ' + output_questions[id]['response'] + '\n'

            # Use best_rationale from self-consistency scoring instead of all_rationales[0]
            best_rationale = output_rationales[id].get('best_rationale', '')
            output_rationale = 'Reasoning: ' + (best_rationale if best_rationale else '') + '\n'

            answer = 'Answer: ' + problems[qid]['choices'][problems[qid]['answer']] + '\n'
            choices = problems[qid]['choices'].copy()
            choices.remove(problems[qid]['choices'][problems[qid]['answer']])
            distractors = ''
            for i, choice in enumerate(choices):
                distractors += f'({i+1}) {choice}\n'
            id += 1

            if not problems[qid]["relevant_question"]:
                prefix = ''
                qg_example, rg_example, dg_example = '', '', ''
            else:
                prefix = 'Refer to the following example, '
                most_relevant_question = problems[qid]["relevant_question"][0]
                qg_example, rg_example, dg_example = build_example(
                    problems[most_relevant_question], most_relevant_question, data_root
                )

            if problems[qid]['image'] != None:
                image = 'Picture: <img>' + \
                        os.path.join(data_root, problems[qid]['split'], str(qid), problems[qid]['image']) + f'</img>\n'
                context = f'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint'] != '' else ''
                prompt = ('based on the above picture and question with its answer obtained through reasoning, '
                          'generate exactly 3 plausible yet incorrect answers which should be similar and '
                          'grammatically consistent with the correct answer and seperate them with numbers (1) (2) (3).')
                user_value = image + output_question + output_rationale + answer + prefix[:-2] + ' and ' + prompt + dg_example
                assistant_value = distractors
            else:
                context = 'Context: ' + problems[qid]['hint'] + '\n' if problems[qid]['hint'] != '' else ''
                prompt = ('based on the above question with its answer obtained through reasoning, '
                          'generate exactly 3 plausible yet incorrect answers which should be similar and '
                          'grammatically consistent with the correct answer and seperate them with numbers (1) (2) (3).')
                user_value = output_question + output_rationale + answer + prefix[:-2] + ' and ' + prompt + dg_example
                assistant_value = distractors

            conversations = [
                {
                    'from': 'user',
                    'value': user_value
                },
                {
                    'from': 'assistant',
                    'value': assistant_value
                }
            ]
            example_dict.update({'conversations': conversations})
            save_list.append(example_dict)

    with open(args.output, 'w') as fp:
        json.dump(save_list, fp)

    print(f"Saved {len(save_list)} examples to {args.output}")


if __name__ == '__main__':
    main()