#!/usr/bin/env bash

  python  bert_pos.py\
    --task_name="POS"  \
    --do_lower_case=False \
    --crf=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=220   \
    --train_batch_size=16   \
    --learning_rate=2e-5   \
    --num_train_epochs=4.0   \
    --output_dir=./output/result_dir


perl conlleval.pl -o '[SEP]' -r -d '\t' < ./output/result_dir/label_test.txt
