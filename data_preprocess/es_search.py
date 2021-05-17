from typing import Dict, List

from elasticsearch import Elasticsearch
import re


class EsHit:
    def __init__(self, score: float, position: int, text: str, type: str):
        """
        Basic information about an ElasticSearch Hit
        :param score: score returned by the query
        :param position: position in the retrieved results (before any filters are applied)
        :param text: retrieved sentence
        :param type: type of the hit in the index (by default, only documents of type "sentence"
        will be retrieved from the index)
        """
        self.score = score
        self.position = position
        self.text = text
        self.type = type


class EsSearch():
    def __init__(self,
                 es_client: str = "localhost",
                 indices: str = "arc_corpus",
                 max_question_length: int = 1000,
                 max_hits_retrieved: int = 500,
                 max_hit_length: int = 300,
                 max_hits_per_choice: int = 100):
        """
        Class to search over the text corpus using ElasticSearch
        :param es_client: Location of the ElasticSearch service
        :param indices: Comma-separated list of indices to search over
        :param max_question_length: Max number of characters used from the question for the
        query (for efficiency)
        :param max_hits_retrieved: Max number of hits requested from ElasticSearch
        :param max_hit_length: Max number of characters for accepted hits
        :param max_hits_per_choice: Max number of hits returned per answer choice
        """
        self._es = Elasticsearch([es_client], retries=3)
        self._indices = indices
        self._max_question_length = max_question_length
        self._max_hits_retrieved = max_hits_retrieved
        self._max_hit_length = max_hit_length
        self._max_hits_per_choice = max_hits_per_choice
        # Regex for negation words used to ignore Lucene results with negation
        self._negation_regexes = [re.compile(r) for r in ["not\\s", "n't\\s", "except\\s"]]


    def get_hits_for_question(self, question: str, choices: List[str]) -> Dict[str, List[EsHit]]:
        """
        :param question: Question text
        :param choices: List of answer choices
        :return: Dictionary of hits per answer choice
        """
        choice_hits = dict()
        for choice in choices:
            choice_hits[choice] = self.filter_hits(self.get_hits_for_choice(question, choice))
        return choice_hits

    # Constructs an ElasticSearch query from the input question and choice
    # Uses the last self._max_question_length characters from the question and requires that the
    # text matches the answer choice and the hit type is a "sentence"
    def construct_qa_query(self, question, choice):
        return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": question[-self._max_question_length:] + " " + choice
                            }}
                        ],
                        "filter": [
                            {"match": {"text": choice}},
                            {"type": {"value": "sentences"}}
                        ]
                    }
                }}

    def construct_hypo_query(self, hypo, para):
        hypo = hypo[-self._max_question_length:]
        hypo = hypo.lower()
        if para is not None:
            para = para.split("@@")
            para = [fact.split(" ") for fact in para]
            para_set = set([word.lower() for fact in para for word in fact])
            # hypo = list(set(hypo.split()).difference(para_set))
            hypo_set = set(hypo.split())
            hypo = list(hypo_set.union(para_set).difference(hypo_set.intersection(para_set)))
            hypo = " ".join(hypo)
        return {"from": 0, "size": self._max_hits_retrieved,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": hypo[-self._max_question_length:]
                            }}
                        ],
                        "filter": [
                            {"type": {"value": "sentences"}}
                        ]
                    }
                }}

    # Retrieve unfiltered hits for input question and answer choice
    def get_hits_for_choice(self, question, para, in_type='question', filter_noise=False):
        if in_type == "question":
            res = self._es.search(index=self._indices, body=self.construct_qa_query(question, para))
        elif in_type == "hypo":
            res = self._es.search(index=self._indices, body=self.construct_hypo_query(question, para))
        hits = []
        for idx, es_hit in enumerate(res['hits']['hits']):
            es_hit = EsHit(score=es_hit["_score"],
                           position=idx,
                           text=es_hit["_source"]["text"],
                           type=es_hit["_type"])
            hits.append(es_hit)
        if filter_noise:
            hits = self.filter_hits(hits)
        return hits

    # Remove hits that contain negation, are too long, are duplicates, are noisy.
    def filter_hits(self, hits: List[EsHit]) -> List[EsHit]:
        filtered_hits = []
        selected_hit_keys = set()
        for hit in hits:
            hit_sentence = hit.text
            hit_sentence = hit_sentence.strip().replace("\n", " ")
            if len(hit_sentence) > self._max_hit_length:
                continue
            for negation_regex in self._negation_regexes:
                if negation_regex.search(hit_sentence):
                    # ignore hit
                    continue
            if self.get_key(hit_sentence) in selected_hit_keys:
                continue
            if not self.is_clean_sentence(hit_sentence):
                continue
            filtered_hits.append(hit)
            selected_hit_keys.add(self.get_key(hit_sentence))
        return filtered_hits[:self._max_hits_per_choice]

    # Check if the sentence is not noisy
    def is_clean_sentence(self, s):
        # must only contain expected characters, should be single-sentence and only uses hyphens
        # for hyphenated words
        return (re.match("^[a-zA-Z0-9][a-zA-Z0-9;:,\(\)%\-\&\.'\"\s]+\.?$", s) and
                not re.match(".*\D\. \D.*", s) and
                not re.match(".*\s\-\s.*", s))

    # Create a de-duplication key for a HIT
    def get_key(self, hit):
        # Ignore characters that do not effect semantics of a sentence and URLs
        return re.sub('[^0-9a-zA-Z\.\-^;&%]+', '', re.sub('http[^ ]+', '', hit)).strip().rstrip(".")
