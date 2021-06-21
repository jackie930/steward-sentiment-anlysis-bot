# albert-chinese-ner

## Resources

- [Bert](https://github.com/google-research/bert)
- [ALBert](https://github.com/albertlauncher/albert)
- [ALBert_zh](https://github.com/brightmart/albert_zh)

## Papers

- [ALBERT](https://arxiv.org/pdf/1909.11942.pdf)

## QUICK START

* prepare the training data from jsonp to bio format 
```
cd data
# note to change the dataset path
python convert_bio.py
```


* Download the pretrained model

```
wget -P ./albert_base_zh https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip
cd ./albert_base_zh
unzip albert_base_zh_additional_36k_steps.zip
```
* train model

```
source activate tensorflow_p36
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
```

the output of checkpoints will be saved in `albert_base_zh`, output serving model in `output` folder

## output

```bash
INFO:tensorflow:  eval_f = 0.9280548
INFO:tensorflow:  eval_precision = 0.923054
INFO:tensorflow:  eval_recall = 0.9331808
INFO:tensorflow:  global_step = 2374
INFO:tensorflow:  loss = 13.210413
```

测试结果同样：

```
[CLS]
B-LOC
I-LOC
O
B-LOC
I-LOC
I-PER
O
O
O
O
O
O
O
O
O
[SEP]
[CLS]
```

