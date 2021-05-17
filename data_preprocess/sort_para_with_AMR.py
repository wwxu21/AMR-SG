import json
import os
import sys
import re
import itertools
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))
from smatch import amr
import argparse

class Graph(object):
    def __init__(self, cons, rels, adj, graph_id=0, ques=None, choice=None, max_path_num=3):
        self.max_path_num = max_path_num
        self.prepare_tokenize(cons, rels, adj)
        if ques is not None:
            self.q_cons = list(set(ques))
        if choice is not None:
            self.c_cons = list(set(choice))
        self.q_anchor = [] #question related anchor
        self.c_anchor = [] #choice related anchor
        self.fact_anchor_dict = {}
        self.fact_con_dict = {}
        # self.graph_dict = adj
        self.graph_id = graph_id
        # self.generate_nodes(cons, rels)
        if ques is not None or choice is not None:
            self.rm_qc_rels()
    def prepare_tokenize(self, cons, rels, adj):
        self.con_encoder = {}
        self.rel_encoder = {}

        self.can_merge = {}
        for i, x in enumerate(cons):
            self.con_encoder[x] = i
        for i, x in enumerate(rels):
            self.rel_encoder[x] = i

        self.graph_dict = {x:{} for x in self.con_encoder}
        reverse_token = "~"
        n_orig = len(self.graph_dict)
        for src in adj:
            if "flag" not in adj[src]:
                self.can_merge[cons[src]] = 1

            for tgt in adj[src]:
                if tgt != "flag": # flag means this node have attribute e.g. concept "name" can have attributes "Donald" "Trump". these kind of nodes are too general
                    self.graph_dict[cons[src]][cons[tgt]] = rels[adj[src][tgt]]
                    self.graph_dict[cons[tgt]][cons[src]] = reverse_token + rels[adj[src][tgt]]
        assert n_orig == len(self.graph_dict)

    def rm_qc_rels(self):
        rm_item = []
        for x in self.graph_dict:
            for y in self.graph_dict[x]:
                if (x in self.q_cons and y in self.c_cons) or (x in self.c_cons and y in self.q_cons):
                    rm_item.append((x,y))
        for x in rm_item:
            self.graph_dict[x[0]].pop(x[1])

    def create_adj(self, candidate_fact):
        n_fact = len(candidate_fact) + 1  # 1 for query
        # assert n_fact <= 12
        adj = [[0] * n_fact for i in range(n_fact)]
        for i in range(n_fact):
            if i >= 1 and candidate_fact[i - 1] not in self.fact_anchor_dict:
                adj[i][i] = 1
                continue
            for j in range(i, n_fact):
                if j >= 1 and candidate_fact[j - 1] not in self.fact_anchor_dict:
                    continue
                if i == j:
                    adj[i][j] = 1
                else:
                    if i == 0:
                        fact1 = set(self.q_anchor).union(self.c_anchor)
                        fact2 = self.fact_anchor_dict[candidate_fact[j - 1]]
                        intersection = list(fact1.intersection(fact2))
                        if len(intersection) != 0:
                            adj[i][j] = 1
                            adj[j][i] = 1
                    else:
                        fact1 = set(self.fact_anchor_dict[candidate_fact[i - 1]])
                        fact2 = self.fact_anchor_dict[candidate_fact[j - 1]]
                        intersection = list(fact1.intersection(fact2))
                        if len(intersection) != 0:
                            adj[i][j] = 1
                            adj[j][i] = 1
        return adj

    def merge(self, graph_list):
        can_merge = []
        cant_merge = []
        for fact_graph in graph_list:
            if len(set(self.can_merge).intersection(fact_graph.can_merge)) != 0:
                can_merge.append(fact_graph)
            else:
                cant_merge.append(fact_graph)
        for fact_graph in can_merge:
            self.merge_one(fact_graph)
        return cant_merge

    def merge_one(self, fact_graph):
        anchors = list(set(self.can_merge).intersection(fact_graph.can_merge))
        con_list = list(set(self.can_merge).union(fact_graph.can_merge))

        for x in anchors:
            if x in self.q_cons and x not in self.q_anchor:
                self.q_anchor.append(x)
            elif x in self.c_cons and x not in self.c_anchor:
                self.c_anchor.append(x)
            for y in self.fact_con_dict:
                if x in self.fact_con_dict[y] and x not in self.fact_anchor_dict[y]:
                    self.fact_anchor_dict[y].append(x)

        self.fact_con_dict[fact_graph.graph_id] = list(fact_graph.con_encoder)
        self.fact_anchor_dict[fact_graph.graph_id] = anchors
        con_count = len(self.con_encoder)
        rel_count = len(self.rel_encoder)

        self.can_merge.update({x: 1 for x in con_list})
        for x in fact_graph.con_encoder:
            if x not in self.con_encoder:
                self.con_encoder[x] = con_count
                con_count += 1
        for x in fact_graph.rel_encoder:
            if x not in self.rel_encoder:
                self.rel_encoder[x] = rel_count
                rel_count += 1

        for src in fact_graph.graph_dict:
            if src not in self.graph_dict:
                self.graph_dict[src] = fact_graph.graph_dict[src]
            else:
                for tgt in fact_graph.graph_dict[src]:
                    if tgt not in self.graph_dict[src]:
                        self.graph_dict[src][tgt] = fact_graph.graph_dict[src][tgt]

    def sample(self):
        max_count = self.max_path_num
        Exist_path = True
        candidates = []
        if len(self.q_anchor) == 0:
            Exist_path = False
        if len(self.c_anchor) == 0:
            Exist_path = False

        if Exist_path:
            graph_connect = {}
            target = [y for y in self.fact_anchor_dict if len(set(self.fact_anchor_dict[y]).intersection(self.q_anchor)) != 0 ]

            graph_connect['q'] = target

            for y in self.fact_anchor_dict:
                target = [z for z in self.fact_anchor_dict if z != y and z not in graph_connect['q'] and len(set(self.fact_anchor_dict[y]).intersection(self.fact_anchor_dict[z])) != 0 ]
                if len(set(self.fact_anchor_dict[y]).intersection(self.c_anchor)) != 0:
                    target.append('c')
                graph_connect[y] = target

            candidates = DFS_search('q', 'c', graph_connect)
        return candidates

    def sample_inside(self, path):
        inside_path_list = []
        for i in range(len(path)):
            if path[i] == "q":
                continue
            if path[i] == "c":
                break
            if path[i-1] == "q":
                start = list(set(self.q_anchor).intersection(self.fact_anchor_dict[path[i]]))
                assert  len(start) != 0
            else:
                start = list(set(self.fact_anchor_dict[path[i-1]]).intersection(self.fact_anchor_dict[path[i]]))
            if path[i+1] == 'c':
                end = list(set(self.c_anchor).intersection(self.fact_anchor_dict[path[i]]))
                assert  len(end) != 0
            else:
                end = list(set(self.fact_anchor_dict[path[i+1]]).intersection(self.fact_anchor_dict[path[i]]))

            inside_path = []
            for x in start:
                for y in end:
                    rels_dict = {}
                    rels_dict[x] = [q for q in self.graph_dict[x] if q in self.fact_con_dict[path[i]]]
                    for p in self.fact_con_dict[path[i]]:
                        if p != x and p != y:
                            target = [q for q in self.graph_dict[p] if q in self.fact_con_dict[path[i]] and q not in rels_dict[x] and q != x]
                            if end in target:
                                target.remove(end)
                                target.append(end)
                            rels_dict.update({p:target})
                    inside_path.extend(DFS_search(x, y, rels_dict))
            inside_path_list.append(inside_path)
        inside_path_combo = list(itertools.product(*inside_path_list))
        out = []
        for combo in inside_path_combo:
            for i_p, part in enumerate(combo):
                if i_p < len(combo) - 1 and combo[i_p][-1] != combo[i_p+1][0]:
                    break
            else:
                out.append([combo[i][j] for i in range(len(combo)) for j in range(len(combo[i])) if i == 0 or j != 0])

        return out

def DFS_search(start, end, rels_input):
    if isinstance(rels_input[start], list):
        rels = rels_input
    visit = [start]
    path = [start]
    stack = []
    start_depth = [x for x in rels[start] if x in rels]
    stack.extend(start_depth)
    paths = []
    candidate = {}
    while (len(stack) != 0):
        curr_node = stack.pop(-1)
        if curr_node != end and curr_node not in rels and curr_node != "flag" and curr_node != end and curr_node != start:
            continue
        if curr_node in visit:
            continue
        if curr_node == end or curr_node in candidate:
            paths.append([x for x in path] + [end])
            for x in path:
                if x != 'q':
                    candidate[x] = 1
            continue
        if curr_node == "flag":
            path.pop(-1)
            continue

        visit.append(curr_node)
        path.append(curr_node)
        stack.append("flag")
        if any([x in candidate for x in rels[curr_node] if x not in stack]):
            candidate[curr_node] = 1
        next_depth = [x  for x in rels[curr_node] if x not in stack and x not in visit]

        stack.extend(next_depth)

    return list(candidate)

def sample_path(con_one, rel_one, adj_one, max_path_num=1):
    """
    all inputs are after tokenization and in ids form. it is compatiable to merge the last two dimension (W,S),
    which is used to identify a word
    :param con_one: type: list, size: C*N*W,  C:choice number,
    N: fact number, the first fact is the query, the remaining are the para, W: word number for each fact concept
    :param rel_one: type: list, size: C*N*W*S,  C:choice number,
    N: fact number, the first fact is the query, the remaining are the para, W: word number for each fact relation
    :param adj_one: type: list of dict, size: C*(Dict of Dict). (Dict of Dict) is the adjacent dict, key is the source concept id,
    value is also a dict, where the sub dict key is target concept id, value is the relation id
    :param max_path_num: sample path number (not used)
    :return: type: list, size: C*L, C:choice number, L: sampled fact ids in the path
    """
    prefix = set([1, 2, 3, 4, 5]) # avoid empty
    global max_fact

    query_cons = [x[0] for x in con_one]
    assert len(query_cons) != 0
    ques_cons = []
    for x in query_cons[0]:
        flag = [x in query_con for query_con in query_cons]
        if len(flag) == sum(flag):
            ques_cons.append(x)
    output = []
    for i_c in range(len(con_one)):
        con_choice = con_one[i_c]
        rel_choice = rel_one[i_c]
        adj_choice = adj_one[i_c]
        choice_cons = [x for x in con_choice[0] if x not in ques_cons]
        # create a larger graph based on all facts
        query_graph = Graph(con_choice[0], rel_choice[0], adj_choice[0], graph_id=0,ques=ques_cons, choice=choice_cons, max_path_num=max_path_num)
        fact_graphs = [Graph(con_choice[x], rel_choice[x], adj_choice[x], graph_id=x, max_path_num=max_path_num) for x in range(len(con_choice)) if x != 0 and con_choice[x] != []]
        last_size = len(fact_graphs)
        while True:
            fact_graphs = query_graph.merge(fact_graphs)
            remain_size = len(fact_graphs)
            if remain_size == 0 or remain_size == last_size:
                break
            else:
                last_size = remain_size
        candidate_fact = query_graph.sample() # return all past fact ids
        candidate_fact = sorted(list(prefix.union(candidate_fact)))
        if len(candidate_fact) >= max_fact:
            new_candidate_fact = candidate_fact[:max_fact]
        else:
            gap = max_fact - len(candidate_fact)
            new_candidate_fact = []
            for idx in range(1, max_fact + 1):
                if gap - len(new_candidate_fact) == 0:
                    break
                if idx not in candidate_fact:
                    new_candidate_fact.append(idx)
            new_candidate_fact = sorted(list(set(candidate_fact + new_candidate_fact)))
        fact_adj = query_graph.create_adj(new_candidate_fact)
        output.append((new_candidate_fact, fact_adj))

    return output

def preprocess_AMR(amr_raw):
    gcls_token, gclsr_token, rgclsr_token = "[gcls]", "[gclsr]", "~[gclsr]"
    #prepare concept tokens
    def clean_token(tok):
        if re.sub("[0-9]*", "", tok) == "":
            return tok
        else:
            tok = re.sub("[0-9]*", "", tok)
            tok = tok.replace("-", " ").strip()
            tok = tok.replace("_", "")
            return tok

    n_cons = len(amr_raw.node_values)
    cons_dict = {}
    cons_count = 0
    for x in amr_raw.node_values:
        con_token = clean_token(x)
        if con_token not in cons_dict:
            cons_dict[con_token] = cons_count
            cons_count += 1
    for src in amr_raw.attributes:
        for x in src:
            if len(x) > 0 and x[0] != 'TOP' and x[0] != "<UNK>" and "wiki" not in x[0] and clean_token(x[1]) not in cons_dict:
                con_token = clean_token(x[1])
                cons_dict[con_token] = cons_count
                cons_count += 1

    rel_name_dict = {}
    count = len(rel_name_dict)
    src_dict = {}
    for i in range(n_cons):
        tgt_dict = {}
        new_i = cons_dict[clean_token(amr_raw.node_values[i])]
        for tgt in amr_raw.relations[i]:
            if len(tgt) != 0:
                rel_type = clean_token(tgt[0])
                j = amr_raw.nodes.index(tgt[1])
                new_j = cons_dict[clean_token(amr_raw.node_values[j])]
                if rel_type not in rel_name_dict:
                    rel_name_dict[rel_type] = count
                    count += 1
                tgt_dict[new_j] = rel_name_dict[rel_type]

        tgt_list = amr_raw.attributes[i]
        flag = False
        for tgt in tgt_list:
            if len(tgt) > 0 and tgt[0] != 'TOP' and "wiki" not in tgt[0] and tgt[0] != "<UNK>":
                flag = True
                rel_type = clean_token(tgt[0])
                con_token = clean_token(tgt[1])
                new_j = cons_dict[con_token]
                if rel_type not in rel_name_dict:
                    rel_name_dict[rel_type] = count
                    count += 1
                tgt_dict[new_j] = rel_name_dict[rel_type]
        if flag:
            tgt_dict['flag'] = 1
        if new_i not in src_dict:
            src_dict[new_i] = tgt_dict
        else:
            src_dict[new_i].update(tgt_dict)

    n_count = len(src_dict)
    n_total = len(cons_dict)
    for i in range(n_count, n_total):
        src_dict[i] = {}
    cons = [x for x in cons_dict]
    rels = [x for x in rel_name_dict]

    assert len(cons) == len(src_dict)
    return [cons, rels, src_dict]

def retrieve_para(candidates, fact_info, amr_info):
    global max_fact # 1 for query
    facts_id = [c[0] for c in candidates]
    fact_adjs = [c[1] for c in candidates]
    candidate_facts = [[fact_info[i_c][f] for f in c if f < len(fact_info[i_c])] for i_c, c in enumerate(facts_id)]
    candidate_amrs = [[amr_info[i_c][f] for f in c if f < len(amr_info[i_c])] for i_c, c in enumerate(facts_id)]
    fact_adjs = [[row + [0] * (max_fact + 1 - len(row)) for row in c] + [[0] * (max_fact + 1)] * (
                max_fact + 1 - len(c)) for c in fact_adjs]
    return candidate_facts, candidate_amrs, fact_adjs

def sort_one(example):
    choices = example['question']['choices']
    amr_info = []
    fact_info = []
    parsed_amr_info = []
    for i_choice, choice in enumerate(choices):
        hypo = choice['hypo']
        paras = choice['para'].split("@@")
        paras = [x for x in paras if x != ""]
        fact_info_one = [hypo] + paras

        hypo_amr = choice['hypo-amr']
        paras_amr = choice['paras_amr'].split("@@")
        paras_amr = [x for x in paras_amr if x != ""]
        amr_info_one = [hypo_amr] + paras_amr


        hypo_parsed_amr = amr.AMR.parse_AMR_line(hypo_amr)
        hypo_parsed_amr_info = preprocess_AMR(hypo_parsed_amr)
        paras_parsed_amr = [amr.AMR.parse_AMR_line(x) for x in paras_amr if x != ""]
        paras_parsed_amr_info = [preprocess_AMR(x) for x in paras_parsed_amr]
        parsed_amr_info_one = [hypo_parsed_amr_info] + paras_parsed_amr_info

        assert len(amr_info_one) == len(fact_info_one)

        assert len(parsed_amr_info_one) == len(fact_info_one)

        fact_info.append(fact_info_one)
        amr_info.append(amr_info_one)
        parsed_amr_info.append(parsed_amr_info_one)

    cons = [[f[0] for f in c] for c in parsed_amr_info]
    rels = [[f[1] for f in c] for c in parsed_amr_info]
    adjs = [[f[2] for f in c] for c in parsed_amr_info]

    candidates = sample_path(cons, rels, adjs, max_path_num=1)

    candidate_facts, candidate_amrs, fact_adjs = retrieve_para(candidates, fact_info, amr_info)
    para = ["@@".join(x) for x in candidate_facts]
    paras_amr = ["@@".join(x) for x in candidate_amrs]
    c_lst_AMR = []
    for i in range(len(para)):
        c_lst_AMR.append({"text": choices[i]['text'],
                          "label": choices[i]['label'],
                          'hypo': choices[i]['hypo'],
                          'hypo-amr': choices[i]['hypo-amr'],
                          'para': para[i],
                          'paras_amr': paras_amr[i],
                          "paras_adj": fact_adjs[i]})

    output_dict_AMR = {
        "id": example["id"],
        "question": {
            "stem": example["question"]["stem"],
            "choices": c_lst_AMR,
        },
        "fact1": example['fact1'],
        "answerKey": example["answerKey"],
    }
    return output_dict_AMR

def sort_with_AMR(qa_file, output_file):
    with open(output_file, 'w') as writer1, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        line_tqdm = tqdm(enumerate(qa_handle), dynamic_ncols=True)
        for i_l, line in line_tqdm:
            json_line = json.loads(line)
            output_dict_AMR = sort_one(json_line)
            writer1.write(json.dumps(output_dict_AMR) + "\n")

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
    args.output = args.file + '_preprocess'
    args.input = args.file + '2'
    return args

#adapted by weiwen
if __name__ == "__main__":
    max_fact = 15
    args = get_args()
    train_qa_file = os.path.join(args.input, 'train.jsonl')
    test_qa_file = os.path.join(args.input, 'test.jsonl')
    dev_qa_file = os.path.join(args.input, 'dev.jsonl')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train_output_file = os.path.join(args.output, 'train.jsonl')
    test_output_file = os.path.join(args.output, 'test.jsonl')
    dev_output_file = os.path.join(args.output, 'dev.jsonl')

    sort_with_AMR(train_qa_file, train_output_file)
    sort_with_AMR(test_qa_file, test_output_file)
    sort_with_AMR(dev_qa_file, dev_output_file)



