from elasticsearch import Elasticsearch
import argparse

def store_corpus(task_name,input_file):
    es = Elasticsearch()
    index = task_name+'_corpus'
    with open(input_file,'rt',encoding='utf-8') as fin:
        for i, line in enumerate(fin, 1):
            sentence = line.strip().replace('"','')
            sentence = sentence.lower()
            sentence = {
                "text": sentence
            }
            resp = es.index(index=index, doc_type="sentences", id=i, body=sentence)
            if i % 100 == 0:
                print(i, " sentences has been stored!!")
    print("Finish!!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='ARC',
        type=str,
        required=False,
        help="The input corpus file. Should contain the .txt files for the task.",
    )

    args = parser.parse_args()
    task_name = args.task_name
    print(task_name)

    if task_name.lower() == 'openbook_comm':
        input_file = '../Data/ARC-14m/ARC_Openbook.txt'

    elif task_name.lower() == 'openbook_core':
        input_file = '../Data/OpenBook/Main/core.txt'

    else:
        raise ValueError('Invalid task name!')

    store_corpus(task_name,input_file)

if __name__ == "__main__":
    main()
