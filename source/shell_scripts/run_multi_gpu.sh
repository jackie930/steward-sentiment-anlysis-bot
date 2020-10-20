#!/usr/bin/env bash
source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12

# assert gpu setting
cuda_devices=(${CUDA_VISIBLE_DEVICES//,/ })
if [ ${num_gpu_cores} == ${#cuda_devices[@]} ]
then
    echo "cuda devices and gpu cores matched"
else
    echo "error: please reset cuda devices or num_gpu_cores..."
    exit 1
fi

# set environmental variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# execute the task
echo "task starts..."

python ./bert/run_classifier_multi_gpu.py \
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
  --weight_list='1,1,1'
  ---use_gpu=true \
  --num_gpu_cores=4

echo "task is done..."