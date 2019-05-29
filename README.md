# BERT POS tagger (tensorflow)

Fine-tune Google's BERT for POS-tagging task (UD Treebank as the dataset). 

## Folder Description:
```
bert-tensorflow-pos
|____ pos_tagger_bert_tensorflow.ipynb  # Notebook with all actions required to download UD Treebank dataset and BERT model, fine-tune BERT, and evaluate POS tagger
|____ bert_pos.py           # Main code
|____ data                  # Train data
|____ middle_data           # Middle data (label id map)
|____ output                # Output (final model, predicted results)
|____ uncased_L-12_H-768_A-12	# BERT model downloaded from -> https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
|____ run_pos.sh    		# Run model and evaluate results

```

## Install packages:
```
pip install pyconll  # for UD treebank reading
pip install bert-tensorflow # for using bert model
```

## Usage:
Preferably run notebook `pos_tagger_bert_tensorflow.ipynb`, or
```
bash run_pos.sh
```

## What's in run_pos.sh:
```
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
```

## References:

[1] https://arxiv.org/abs/1810.04805

[2] https://github.com/google-research/bert

