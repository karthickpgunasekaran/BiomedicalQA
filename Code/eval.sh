#!/bin/bash

​//This batch scripts helps in validation/testing of model where in each checkpoint during training is used to make predictions

export SQUAD_DIR=path-to-data-directory



echo ' Started !'



i=500
//For each saved model during training, 

while [[ $i -lt 5001 ]]
do
    export OUTPUT_DIR=path-to-saving-predictions/$i/
    
​    export BIOBERT_DIR =path-to-model/
    
    echo "Training model checkpoint: $i"
    python Yesno_BioQA.py \
	--do_train=False \
	--do_predict=True \
	--vocab_file=path-to-model/vocab.txt \
	--bert_config_file=path-to-model/bert_config.json \
	--init_checkpoint=path-to-model/model.ckpt-$i \ 
	--max_seq_length=384 \
	--train_batch_size=10 \
	--learning_rate=5e-6 \
	--evaluation=False \
	--doc_stride=128 \
	--do_lower_case=False \
	--num_train_epochs=2 \
	--train_file=path-to-train-file/train_file.json \
	--predict_file=path-to-test-file/test_file.json \
	--output_dir=$OUTPUT_DIR
    i=$((i+500))
done

echo 'karthick: All Done!'

