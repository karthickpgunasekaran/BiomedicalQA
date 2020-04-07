#!/bin/bash
​
export SQUAD_DIR=path-to-data-directory/
export OUTPUT_DIR=path-to-output-directory/

​export BIOBERT_DIR =path-to-model/

python Yesno_BioQA.py \
	--do_train=True \
	--do_predict=False \
	--vocab_file=path-to-model/vocab.txt \
	--bert_config_file=path-to-model/bert_config.json \
	--init_checkpoint=path-to-model/bert_model.ckpt \
	--max_seq_length=384 \
	--train_batch_size=2 \
	--learning_rate=5e-6 \
	--doc_stride=128 \
	--evaluation=False \
	--do_lower_case=False \
	--num_train_epochs=10 \
	--train_file=path-to-train-file/train_file.json \
	--predict_file=path-to-test-file/test_file.json \
	--output_dir=$OUTPUT_DIR
