from tqdm import tqdm
import json
import argparse
import os

def read_openbook(addr):
    with open(addr, 'r') as reader:
        ob = reader.readlines()
        ob = [x.strip().replace('"', '') for x in ob]
    return ob

def read_qa(addr):
    qa = []
    with  open(addr, 'r') as reader:
        qa_tqdm = tqdm(reader, dynamic_ncols=True)
        for line in qa_tqdm:
            json_line = json.loads(line)
            qa.append(json_line)
    return qa

def read_hypo(addr):
    with open(addr, 'r') as reader:
        hypos = reader.readlines()
    return hypos

def enhance_hypo(qa, hypos, corpus='openbook'):
    hypos = read_hypo(hypos)
    total_choices_num = 0
    out_qa = []
    for i_line, line in enumerate(qa):
        num_choice = len(line['question']['choices'])
        hypo = hypos[total_choices_num: total_choices_num + num_choice]
        total_choices_num += num_choice
        hypo = [x.strip() for x in hypo]
        c_lst = []
        for i in range(len(hypo)):
            c_lst.append({"text": line["question"]['choices'][i]['text'], "label": line["question"]['choices'][i]['label'], 'hypo': hypo[i]})
        if corpus == "openbook":
            output_dict = {
                "id": line["id"],
                "question": {
                    "stem": line["question"]["stem"],
                    "choices": c_lst,
                },
                "fact1": line["fact1"],
                "answerKey": line["answerKey"],
            }
        elif corpus == 'arc':
            output_dict = {
                "id": line["id"],
                "question": {
                    "stem": line["question"]["stem"],
                    "choices": c_lst,
                },
                "answerKey": line["answerKey"],
            }
        out_qa.append(output_dict)
    return out_qa

def read_amr(addr, start=6):
    if addr is None:
        return {}
    amrs = []
    ids = []
    id_printed = []
    snts = []
    begin = False
    with open(addr, 'r') as reader:
        for each in reader:
            if each.startswith("# ::id"):
                begin = True
                idx = each.strip().split(' ')[2]
                ids.append(idx)
                amr = []
            if each.startswith("# ::snt"):
                snt = each.strip()[8:].strip()
                snts.append(snt)
            if each.strip() == '':
                begin = False
                id_printed.append(idx)
                amrs.append(amr)
            if begin:
                amr.append(each.strip())
    amrs = [amr[start:] for amr in amrs]
    amrs = [''.join(amr) for amr in amrs]
    amr_dict = {snts[i_a]: amrs[i_a] for i_a, amr in enumerate(amrs)}
    return amr_dict

def enhance_amr_fact(amrs, save):
    amrs = read_amr(amrs, start=6)
    with open(save, 'w') as writer:
        for i_line, line in enumerate(amrs):
            amr = amrs[line]
            output_dict = {
                "id": i_line,
                "text": line,
                "amr": amr,
            }
            writer.write(json.dumps(output_dict) + "\n")


def enhance_amr(qa, amr_addr, save):
    amr = read_amr(amr_addr)
    with open(save, 'w') as writer:
        for i_line, line in enumerate(qa):
            choices = line['question']['choices']
            hypos = [x['hypo'] for x in choices]
            amr4 = [amr[h] for h in hypos]
            c_lst = []
            for i in range(len(amr4)):
                c_lst.append({"text": line["question"]['choices'][i]['text'],
                              "label": line["question"]['choices'][i]['label'],
                              'hypo': line["question"]['choices'][i]['hypo'],
                              'hypo-amr':amr4[i],})
            output_dict = {
                "id": line["id"],
                "question": {
                    "stem": line["question"]["stem"],
                    "choices": c_lst,
                },
                "fact1": line["fact1"],
                "answerKey": line["answerKey"],
            }
            writer.write(json.dumps(output_dict) + "\n")

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
    args.output = args.file + '0'
    args.input = args.file
    return args


if __name__ == "__main__":
    args = get_args()
    train_qa_file = os.path.join(args.input, 'train.jsonl')
    test_qa_file = os.path.join(args.input, 'test.jsonl')
    dev_qa_file = os.path.join(args.input, 'dev.jsonl')
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_hypo_file = os.path.join(args.input, 'train-hypo.txt')
    test_hypo_file = os.path.join(args.input, 'test-hypo.txt')
    dev_hypo_file = os.path.join(args.input, 'dev-hypo.txt')

    train_amr_file = os.path.join(args.input, 'train-amr.txt')
    test_amr_file = os.path.join(args.input, 'test-amr.txt')
    dev_amr_file = os.path.join(args.input, 'dev-amr.txt')
    core_amr_file = os.path.join(args.input, 'core-amr.txt')
    comm_amr_file = os.path.join(args.input, 'comm-amr.txt')

    train_output_file = os.path.join(args.output, 'train.jsonl')
    test_output_file = os.path.join(args.output, 'test.jsonl')
    dev_output_file = os.path.join(args.output, 'dev.jsonl')
    core_output_file = os.path.join(args.file, 'core.jsonl')
    comm_output_file = os.path.join(args.file, 'comm.jsonl')

    train_qa_file = read_qa(train_qa_file)
    test_qa_file = read_qa(test_qa_file)
    dev_qa_file = read_qa(dev_qa_file)
    # add hypo to the dataset
    train_qa_file = enhance_hypo(train_qa_file, train_hypo_file)
    test_qa_file = enhance_hypo(test_qa_file, test_hypo_file)
    dev_qa_file = enhance_hypo(dev_qa_file, dev_hypo_file)
    # add amr to the dataset
    enhance_amr(train_qa_file, train_amr_file, train_output_file)
    enhance_amr(dev_qa_file, dev_amr_file, dev_output_file)
    enhance_amr(test_qa_file, test_amr_file, test_output_file)
    enhance_amr_fact(core_amr_file, core_output_file)
    enhance_amr_fact(comm_amr_file, comm_output_file)
