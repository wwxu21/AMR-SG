"""
Script to convert the retrieved hits into a paragraph comprehension dataset. Questions with no
hits are mapped to a blank paragraph.
USAGE:
 python get_ob_context.py

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
from typing import List, Dict

# from allennlp.common.util import JsonDict
from tqdm import tqdm
# from extract import AMRIO
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from es_search import EsSearch, EsHit
import argparse
# from smatch import smatch, amr



max_sentences_per_choice = 100
es_search_arc_corpus = EsSearch(indices= 'openbook_comm_corpus', max_hits_per_choice=max_sentences_per_choice, max_hits_retrieved=max_sentences_per_choice)
es_search_per_question = EsSearch(indices= 'openbook_core_corpus', max_hits_per_choice=30, max_hits_retrieved=100)
global_max_length = 0

def read_amr(in_file, punc=False):
    with open(in_file, 'r') as reader:
        out_dict = {}
        for line in reader:
            ob_line = json.loads(line)
            if punc and (not ob_line['text'].endswith(".")):
                ob_text = ob_line['text'] + ' .'
            else:
                ob_text = ob_line['text']
            out_dict[ob_text.lower()] = ob_line['amr']
    # modification_list = []
    # line_tqdm = tqdm(enumerate(out_dict), dynamic_ncols=True)
    # for i_x, x in line_tqdm:
    #     try:
    #         parsed_amr = amr.AMR.parse_AMR_line(out_dict[x])
    #         if parsed_amr is None:
    #             modification_list.append(i_x + 1)
    #     except:
    #         modification_list.append(i_x + 1)
    # print(modification_list)
    # assert len(modification_list) == 0
    return out_dict

def add_retrieved_text(qa_file, output_file, core_file, comm_file, func):
    core_dict = read_amr(core_file, punc=True)
    comm_dict = read_amr(comm_file, punc= False)
    with open(output_file, 'w') as writer1, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        line_tqdm = tqdm(enumerate(qa_handle), dynamic_ncols=True)
        for i_l, line in line_tqdm:
            json_line = json.loads(line)
            output_dict_AMR = func(json_line, core_dict, comm_dict)
            writer1.write(json.dumps(output_dict_AMR) + "\n")


def add_hits_to_qajson_by_hypo(qa_json, core_dict, comm_dict):
    global  global_max_length
    gold = qa_json['fact1']
    anskey = ord(qa_json['answerKey']) - ord("A")
    if not gold.endswith("."):
        gold = gold + ' .'
        gold = gold.lower()
    question_text = qa_json["question"]["stem"]
    choices_text = [choice["text"] for choice in qa_json["question"]["choices"]]
    hypos_text = [choice["hypo"] for choice in qa_json["question"]["choices"]]
    para_text = [choice["para"] for choice in qa_json["question"]["choices"]]
    hits_per_choice = {choices_text[i]: es_search_arc_corpus.get_hits_for_choice(hypos_text[i], None, in_type='hypo', filter_noise=True) for i,_ in enumerate(hypos_text)}

    choices = qa_json["question"]["choices"]
    choices_hits = [hits_per_choice[choice["text"]] for choice in choices]
    choices_hits = [list(hits[:max_sentences_per_choice]) for hits in choices_hits]
    paras_new = [[hit.text for hit in choice_hits] for choice_hits in choices_hits]

    core_AMR = [[core_dict[x] if x!= "" else "" for x in para.split("@@")] for para in para_text]

    paras_new = [[x for x in para] for para in paras_new]
    comm_AMR =  [[comm_dict[x] for x in para] for para in paras_new]

    core_AMR = ['@@'.join(x) for x in core_AMR]
    comm_AMR = ['@@'.join(x) for x in comm_AMR]
    paras_amr = [core_AMR[i] + "@@" + comm_AMR[i] if core_AMR[i] != "" else comm_AMR[i] for i in range(len(core_AMR))]
    paras_new = ['@@'.join(x) for x in paras_new]
    para = [para_text[i] + "@@" + paras_new[i] if para_text[i] != "" else paras_new[i] for i in range(len(para_text))]
    c_lst_AMR = []
    for i in range(len(paras_new)):
        c_lst_AMR.append({"text": choices[i]['text'],
                      "label": choices[i]['label'],
                      'hypo': choices[i]['hypo'],
                      'hypo-amr': choices[i]['hypo-amr'],
                      'para': para[i],
                      'paras_amr':paras_amr[i]})

    output_dict_AMR = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choices": c_lst_AMR,
        },
        "fact1": gold,
        "answerKey": qa_json["answerKey"],
    }
    return  output_dict_AMR

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
    args.output = args.file + '2'
    args.input = args.file + '1'
    return args

if __name__ == "__main__":
    args = get_args()
    train_qa_file = os.path.join(args.input, 'train.jsonl')
    test_qa_file = os.path.join(args.input, 'test.jsonl')
    dev_qa_file = os.path.join(args.input, 'dev.jsonl')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train_output_file = os.path.join(args.output, 'train.jsonl')
    test_output_file = os.path.join(args.output, 'test.jsonl')
    dev_output_file = os.path.join(args.output, 'dev.jsonl')

    core_file = os.path.join(args.file, 'core.jsonl')
    comm_file = os.path.join(args.file, 'comm.jsonl')
    add_retrieved_text(train_qa_file, train_output_file, core_file, comm_file, func=add_hits_to_qajson_by_hypo)
    add_retrieved_text(test_qa_file, test_output_file, core_file, comm_file, func=add_hits_to_qajson_by_hypo)
    add_retrieved_text(dev_qa_file, dev_output_file, core_file, comm_file, func=add_hits_to_qajson_by_hypo)



