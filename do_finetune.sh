 python3 ./RoBERTa_finetune/finetune.py \
  --task_name openbook --model_name roberta-large --data_path obqa_preprocess \
  --max_len 256 --do_train --do_eval --do_test --learning_rate 2e-5\
  --num_train_epochs 8 --transition_number 2
