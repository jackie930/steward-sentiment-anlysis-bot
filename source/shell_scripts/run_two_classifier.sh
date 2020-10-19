#!/usr/bin/env bash
source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12

python ./bert/run_classifier.py \
  --data_dir='../data' \
  --task_name='chnsenticorp' \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=./output/ \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --learning_rate=5e-5\
  --num_train_epochs=1.0\
  --save_checkpoints_steps=100\
  --weight_list='1,1'

