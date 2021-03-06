# AMR-SG

Code for our **ACL Findings 2021** paper,

**Dynamic Semantic Graph Construction and Reasoning for Explainable Multi-hop Science Question Answering**

Weiwen Xu, Huihui Zhang, Deng Cai and Wai Lam.

## Data Preparation
We present the results on [OpenBookQA](https://leaderboard.allenai.org/open_book_qa/submissions/get-started) and [ARC-Challenge](https://allenai.org/data/arc) in our paper. Due to the license issue, please directly download the datasets from their corresponding websites.

## Data Annotation
1. We use [this repo](https://github.com/kelvinguu/qanli) as our hypothesis generator and [AMR-gs](https://github.com/jcyk/AMR-gs) as our AMR parser. Please follow their instructions to annotate hypothesis and AMR for the datasets respectively.

    Once annotated, please organize the annotated files in the following directory (e.g. OpenBookQA)
    - Data/
        - obqa/
            - train.jsonl (train/dev/test original datasets)
            - dev.jsonl
            - test.jsonl
            - train-hypo.txt (train/dev/test hypotheses)
            - dev-hypo.txt
            - test-hypo.txt
            - train-amr.txt (train/dev/test AMRs)
            - dev-amr.txt
            - test-amr.txt
            - core-amr.txt (core fact AMRs from open-book)
            - comm-amr.txt (common fact AMRs from ARC-Corpus)

2. Create ElasticSearch Server

`python  data_preprocess/store_corpus.py --task_name ${input_file}`

3. Data Preprocessing:

`bash do_preprocess.sh ${input_file}` (e.g. Data/obqa)

4. Retrieve Facts:

`bash do_retrieve.sh ${input_file}`

## Training
`bash do_finetune.sh`

## Citation
If you find this work useful, please star this repo and cite our paper as follows:
```
@inproceedings{xu-etal-2021-dynamic,
    title = "Dynamic Semantic Graph Construction and Reasoning for Explainable Multi-hop Science Question Answering",
    author = "Xu, Weiwen  and
      Zhang, Huihui  and
      Cai, Deng  and
      Lam, Wai",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.90",
    doi = "10.18653/v1/2021.findings-acl.90",
    pages = "1044--1056",
}
```
