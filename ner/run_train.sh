wget -P ./albert_base_zh https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip
cd ./albert_base_zh
unzip albert_base_zh_additional_36k_steps.zip

cd ..
python albert_ner.py \
    --task_name ner \
    --do_train true \
    --do_eval true \
    --data_dir data \
    --vocab_file ./albert_base_zh/vocab.txt \
    --bert_config_file ./albert_base_zh/albert_config_base.json \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --learning_rate 0.01 \
    --num_train_epochs 100 \
    --output_dir albert_base_ner_checkpoints \
    --model_dir='./output' \
    --init_checkpoint albert_base_zh/albert_model.ckpt