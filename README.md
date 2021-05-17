# AMR-SG

Code for our **ACL2020** paper, 

**Dynamic Semantic Graph Construction and Reasoning for Explainable Multi-hop Science Question Answering**

Weiwen Xu, Huihui Zhang, Deng Cai and Wai Lam.

## Data Preparation

## Data Annotation
0. We use [this repo](https://github.com/kelvinguu/qanli) as our hypothesis generator and [AMR-gs](https://github.com/jcyk/AMR-gs) as our AMR parser. Please follow their instructions to annotate hypothesis and AMR for the dataset respectively.
    Once annotated, please organize the annotated files in the following directory (e.g. OpenBookQA)
    - Data/
        - obqa/
            - train.jsonl/ (train/dev/test original datasets)
            - dev.jsonl/
            - test.jsonl/
            - train-hypo.txt/ (train/dev/test hypotheses)
            - dev-hypo.txt/
            - test-hypo.txt/
            - train-amr.txt/ (train/dev/test AMRs)
            - dev-amr.txt/
            - test-amr.txt/
            - core-amr.txt/ (core fact AMRs from open-book)
            - comm-amr.txt/ (common fact AMRs from ARC-Corpus)
1. Create ElasticSearch server `python  data_preprocess/store_corpus.py --task_name ${input_file}`

2. Data Preprocessing: `bash do_preprocess.sh ${input_file}` (e.g. Data/obqa)

3. Retrieve facts: `bash do_retrieve.sh ${input_file}`

## Training

`bash do_finetune.sh`

## Citation
If you find this work useful, please star this repo and cite our paper as follows:
```
@inproceedings{DBLP:conf/emnlp/DengZL20,
  author    = {Yang Deng and
               Wenxuan Zhang and
               Wai Lam},
  title     = {Multi-hop Inference for Question-driven Summarization},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2020, Online, November 16-20, 2020},
  pages     = {6734--6744},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.emnlp-main.547},
}
```
