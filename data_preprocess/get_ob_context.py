"""
Script to convert the retrieved hits into a paragraph comprehension dataset. Questions with no
hits are mapped to a blank paragraph.

JSONL format of files
1. qa_file:
    {
    "id": "9-704",
    "question": {
        "stem": "A flashlight emits",
        "choices": [
                    {"text": "particles", "label": "A"},
                    {"text": "water", "label": "B"},
                    {"text": "bugs", "label": "C"},
                    {"text": "sound", "label": "D"}
                    ]
        },
    "answerKey": "A"
    }

2. output_file:
    {
    "id": "9-704",
    "question": {
        "stem": "A flashlight emits",
        "choices": [
                    {"text": "particles", "label": "A", "para": "Energy stored due to ... charged particles"},
                    {"text": "water", "label": "B", "para": "evaporation is ... moving charged particles"},
                    {"text": "bugs", "label": "C", "para": "stick bugs live on ... charged particles"},
                    {"text": "sound", "label": "D", "para": "vocalizing requires ... articles"}
                    ]
        },
    "answerKey": "A"
    }
"""


import json
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from es_search import EsSearch, EsHit

max_sentences_per_choice = 30
es_search_per_question = EsSearch(indices= 'openbook_core_corpus', max_hits_per_choice=80, max_hits_retrieved=100)


def add_retrieved_text(qa_file, output_file, func):
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        line_tqdm = tqdm(qa_handle, dynamic_ncols=True)
        for line in line_tqdm:
            json_line = json.loads(line)
            output_dict = func(json_line)
            output_handle.write(json.dumps(output_dict) + "\n")

def add_hits_to_qajson_by_hypo(qa_json):
    gold = qa_json['fact1']
    choices_text = [choice["text"] for choice in qa_json["question"]["choices"]]
    hypos_text = [choice["hypo"] for choice in qa_json["question"]["choices"]]
    hits_per_choice = {choices_text[i]: es_search_per_question.get_hits_for_choice(hypos_text[i], None, in_type='hypo', filter_noise=True) for i,_ in enumerate(hypos_text)}
    choices = qa_json["question"]["choices"]
    choices_hits = [hits_per_choice[choice["text"]] for choice in choices]
    choices_hits = [list(hits[:max_sentences_per_choice]) for hits in choices_hits]

    paras = [[hit.text for hit in choice_hits] for choice_hits in choices_hits]
    scores = [[(hit.score, hit.position) for hit in choice_hits] for choice_hits in choices_hits]
    for i_x, x in enumerate(paras):
        for i_y, y in enumerate(x):
            if not y.endswith("."):
                paras[i_x][i_y] = y + ' .'
    paras = ['@@'.join(x) for x in paras]
    c_lst = []
    for i in range(len(paras)):
        c_lst.append({"text": choices[i]['text'],
                      "label": choices[i]['label'],
                      'hypo': choices[i]['hypo'],
                      'hypo-amr': choices[i]['hypo-amr'],
                      'para': paras[i],
                      "scores": scores[i]})
    output_dict = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choices": c_lst,
        },
        "fact1": gold,
        "answerKey": qa_json["answerKey"],
    }
    return output_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default='./Data/obqa',
        type=str,
        required=True,
        help="The original corpus train/dev/test",
    )
    args = parser.parse_args()
    args.output = args.file + '1'
    args.input = args.file + '0'
    return args


if __name__ == "__main__":
    args = get_args()
    train_qa_file =  os.path.join(args.input, 'train.jsonl')
    test_qa_file = os.path.join(args.input, 'test.jsonl')
    dev_qa_file = os.path.join(args.input, 'dev.jsonl')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train_output_file = os.path.join(args.output, 'train.jsonl')
    test_output_file = os.path.join(args.output, 'test.jsonl')
    dev_output_file = os.path.join(args.output, 'dev.jsonl')
    add_retrieved_text(train_qa_file, train_output_file, func=add_hits_to_qajson_by_hypo)
    add_retrieved_text(test_qa_file, test_output_file, func=add_hits_to_qajson_by_hypo)
    add_retrieved_text(dev_qa_file, dev_output_file, func=add_hits_to_qajson_by_hypo)



