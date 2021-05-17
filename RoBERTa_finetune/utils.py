# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from typing import List
import re
import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, hypo=None, hypo_amr=None, context_list=None, context_amr_list=None, context_adj_list=None, endings=None, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            hypo_amr: list of str. The untokenized amr text of query (question + choice)
            context_list: list of list of str. The separate form of contexts
            context_amr_list: list of list of str. The amr of context_list
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.hypo = hypo
        self.hypo_amr = hypo_amr
        self.context_list = context_list
        self.context_amr_list = context_amr_list
        self.context_adj_list = context_adj_list


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 amr_flag
    ):
        self.amr_flag = amr_flag
        self.example_id = example_id
        if amr_flag:
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    "amr_ids": amr_ids,
                    'facts_adj': tokens_b_adj,

                }
                for _, input_ids, input_mask, segment_ids, amr_ids, tokens_b_adj in choices_features
            ]
        else:
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                }
                for _, input_ids, input_mask, segment_ids in choices_features
            ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth))
        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set."""
    def __init__(self, amr_flag):
        super(ArcProcessor, self).__init__()
        self.amr_flag = amr_flag

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        def normalize(truth):
            if truth in "ABCDE":
                return ord(truth) - ord("A")
            elif truth in "12345":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1


            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            question_choices = data_raw["question"]
            options = question_choices["choices"]
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                options.append(question_choices["choices"][2])
            assert truth is not None
            question = question_choices["stem"]
            id = data_raw["id"]
            context_list = [x['para'].split("@@") for x in options]
            context_str = [' '.join(x) for x in context_list]
            hypo = [x['hypo'] for x in options]
            if self.amr_flag:
                hypo_amr = [x['hypo-amr'] for x in options]
                context_amr_list = [x['paras_amr'].split("@@") for x in options]
                context_adj_list = [x['paras_adj'] for x in options]
            else:
                hypo_amr = None
                context_amr_list = None
                context_adj_list = None
            if len(options) == 4 or len(options) == 5:
                examples.append(
                    InputExample(
                        example_id = id,
                        question=question,
                        contexts=[context_str[0].replace("_", ""), context_str[1].replace("_", ""),
                                  context_str[2].replace("_", ""), context_str[3].replace("_", "")],
                        hypo=hypo,
                        hypo_amr=hypo_amr,
                        context_list=context_list,
                        context_amr_list=context_amr_list,
                        context_adj_list=context_adj_list,
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))


        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples

class OpenbookProcessor(DataProcessor):
    """Processor for the OpenBook data set."""
    def __init__(self, amr_flag):
        super(OpenbookProcessor, self).__init__()
        self.amr_flag = amr_flag

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines


    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "0123":
                return int(truth) - 0
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        for line in tqdm.tqdm(lines, desc="read openbook data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            context_list = [x['para'].split("@@") for x in options]
            context_str = [' '.join(x) for x in context_list]
            hypo = [x['hypo'] for x in options]
            if self.amr_flag:
                hypo_amr = [x['hypo-amr'] for x in options]
                context_amr_list = [x['paras_amr'].split("@@") for x in options]
                context_adj_list = [x['paras_adj'] for x in options]
            else:
                hypo_amr = None
                context_amr_list = None
                context_adj_list = None
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id = id,
                        question=question,
                        contexts=[context_str[0].replace("_", ""), context_str[1].replace("_", ""),
                                  context_str[2].replace("_", ""), context_str[3].replace("_", "")],
                        hypo=hypo,
                        hypo_amr=hypo_amr,
                        context_list=context_list,
                        context_amr_list=context_amr_list,
                        context_adj_list=context_adj_list,
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='<s>',
                                 cls_token_segment_id=1,
                                 sep_token='</s>',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=1,
                                 mask_padding_with_zero=True,
                                 ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    amr_flag = False
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        tooLong = False

        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            tokens_b = tokenizer.tokenize(context)
            tokens_a = tokenizer.tokenize(example.hypo[ending_idx])

            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_b, tokens_a, max_length=max_seq_length - special_tokens_count)
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            if pad_on_left:
                tokens = tokens_b + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                tokens += tokens_a + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_a) + 1)
            else:
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))
        if not tooLong:
            features.append(
                InputFeatures(
                    example_id = example.example_id,
                    choices_features = choices_features,
                    label = label,
                    amr_flag = amr_flag,
                )
        )

    return features


def convert_examples_to_features_with_amr(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='<s>',
                                 cls_token_segment_id=1,
                                 sep_token='</s>',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=1,
                                 mask_padding_with_zero=True,
                                 fact_length=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    amr_flag = True
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.context_list, example.endings)):
            tokens_b = []
            amr_ids = []
            tokens_b_by_list = [tokenizer.tokenize(x) for x in context if x != ""]
            amr_ids_by_list = [[i + 1] * len(tokens_b_by_list[i]) for i in range(len(tokens_b_by_list))]
            [tokens_b.extend(x) for x in tokens_b_by_list]
            [amr_ids.extend(x) for x in amr_ids_by_list]
            tokens_a = tokenizer.tokenize(example.hypo[ending_idx])

            tokens_b_adj = example.context_adj_list[ending_idx]

            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_b, tokens_a, max_seq_length - special_tokens_count)
            amr_ids = amr_ids[:len(tokens_b)]

            #add concept_bpe padding
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            amr_ids = [0] * (len(tokens) - 1) + [-1] + amr_ids
            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
                amr_ids = amr_ids + [-1]
            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
                amr_ids = amr_ids + [-1]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
                amr_ids = [-1] + amr_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                amr_ids = [-1] * padding_length + amr_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                amr_ids = amr_ids + [-1] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(amr_ids) == max_seq_length
            amr_ids = expand_amrids(amr_ids, expand_size=fact_length+1) # one for query
            choices_features.append((tokens, input_ids, input_mask, segment_ids, amr_ids, tokens_b_adj))
        label = label_map[example.label]

        features.append(
            InputFeatures(
                example_id = example.example_id,
                choices_features = choices_features,
                label = label,
                amr_flag=amr_flag,
            )
        )

    return features

def expand_amrids(amr_ids, expand_size=16): #one for query
    def expand_one(w, expand_size):
        if isinstance(w[0], list):
            return [expand_one(x, expand_size) for x in w]
        else:
            blank = [[0] * len(w) for i in range(expand_size)]
            for i_x, x in enumerate(w):
                if x == -1:
                    continue
                else:
                    blank[x][i_x] = 1

            return blank

    output = expand_one(amr_ids, expand_size)
    return output

def _truncate_seq_pair(tokens_b, tokens_a, max_length=256, amr_belong=None):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        else:
            tokens_b.pop()
            if amr_belong is not None:
                amr_belong.pop()


processors = {
    "race": RaceProcessor,
    "arc": ArcProcessor,
    "arc_challenge": ArcProcessor,
    "arc_regents": ArcProcessor,
    'openbook':OpenbookProcessor,
}



