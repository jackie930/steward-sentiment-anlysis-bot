python albert_ner.py \
    --task_name ner \
    --do_export true \
    --data_dir data \
    --vocab_file ./albert_config/vocab.txt \
    --bert_config_file ./albert_base_zh/albert_config_base.json \
    --max_seq_length 128 \
    --predict_batch_size 8 \
    --output_dir albert_base_ner_checkpoints \
    --model_dir='./output'

