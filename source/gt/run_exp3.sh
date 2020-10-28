#!/usr/bin/env bash
source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12

# execute the task
echo "task starts..."

python ./bert/run_custom_classifier.py \
    --task_name='gt' \
    --do_lower_case=true \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --save_for_serving=true \
    --data_dir='../experiments/exp3' \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5\
    --num_train_epochs=10.0 \
    --use_gpu=true \
    --num_gpu_cores=8 \
    --use_fp16=false \
    --output_dir='./outputs'

echo "task is done..."

