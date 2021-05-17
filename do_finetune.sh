CUDA_VISIBLE_DEVICES=1 python3 ./RoBERTa_finetune/finetune.py \
  --task_name openbook --model_name roberta-large --data_path obqa_preprocess \
  --max_len 256 --do_train --do_eval --do_test --learning_rate 2e-5 --warmup_ratio 0.06\
  --num_train_epochs 8 --per_gpu_train_batch_size 6 --per_gpu_eval_batch_size 1  --gradient_accumulation_steps 2 \
   --transition_number 2 --fp16
