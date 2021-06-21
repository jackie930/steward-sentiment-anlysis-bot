# steward-sentiment-anlysis-bot
基于Google-Bert,进行 `Sentiment Analysis` 的任务 和 `ner`的部分， 其中`ner`部分代码目录在`/ner`下，的操作指南参考链接（see documentation [here](ner/README.md)）, 并利用`AWS SageMaker`进行模型训练和部署。

## Data
本解决方案使用的数据分为两部分，预训练模型的数据和
* 使用的基础模型是从google发布的bert预训练模型得到的，[模型下载地址](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
* 使用的情感分析数据集是新浪微博的短文本，有10万条评论数据，公开数据已经标注了正负向的情感标注

## Model
模型是end-to-end的二分类模型，[模型论文](https://arxiv.org/abs/1810.04805)

2018年google推出了bert模型，这个模型的性能要远超于以前所使用的模型，总的来说就是很牛。但是训练bert模型是异常昂贵的，对于一般人来说并不需要自己单独训练bert，只需要加载预训练模型，就可以完成相应的任务。

## Features

- [x] **CPU/GPU Support**
- [x] **Multi-GPU Support**: [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) is used to achieve Multi-GPU support for this project, which mirrors vars to distribute across multiple devices and machines. The maximum batch_size for each GPU is almost the same as [bert](https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues). So **global batch_size** depends on how many GPUs there are.
    - Assume: num_train_examples = 32000
    - Situation 1 (multi-gpu): train_batch_size = 8, num_gpu_cores = 4, num_train_epochs = 1
        - global_batch_size = train_batch_size * num_gpu_cores = 32
        - iteration_steps = num_train_examples * num_train_epochs / train_batch_size = 4000
    - Situation 2 (single-gpu): train_batch_size = 32, num_gpu_cores = 1, num_train_epochs = 4
        - global_batch_size = train_batch_size * num_gpu_cores = 32
        - iteration_steps = num_train_examples * num_train_epochs / train_batch_size = 4000
    - Result after training is equivalent between situation 1 and 2 when synchronous update on gradients is applied.
- [x] **SavedModel Support**
- [x] **SageMaker Training/Deploy Support**
- [x] **TFserving Support- SavedModel Export**
- [x] **Unbalanced Dataset Customer Loss Support**
- [x] **Multi-Class Support**
- [x] **Multi-Label Support**

## Dependencies

- Tensorflow
  - tensorflow >= 1.11.0   # CPU Version of TensorFlow.
  - tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow. (Upgrade to 1.14.0 when meets [ImportError: No module named 'tensorflow.python.distribute.cross_device_ops' ](https://github.com/HaoyuHu/bert-multi-gpu/issues/11))
- NVIDIA Collective Communications Library (NCCL)

## Quick Start Guide
### Train
使用`SageMaker BYOC`训练的步骤
* 下载预训练模型，放到`./source/bert/pretrain_model`目录下，模型大小364.20M

```
wget -P ./source/bert/pretrain_model https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
cd ./source/bert/pretrain_model
unzip chinese_L-12_H-768_A-12.zip 
```

## run binary classification 
```

source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12


nohup python -u ./bert/run_classifier.py \
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
  --num_train_epochs=5.0\
  --save_checkpoints_steps=100\
  --weight_list='1,1' > train.log 2>&1 &
```

Shell script is available also (see shell_scripts/run_two_classifier.sh)


## run multi-class classification 

here we use example case three class, you can change by define the class

```

source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12


nohup python -u python bert/run_classifier.py \
  --data_dir='../data' \
  --task_name='GTProcessor' \
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
  
```

Shell script is available also (see shell_scripts/run_all.sh)

## run multi-gpu classification 

here we use example case three class, you can change by define the class

```

source activate tensorflow_p36
export BERT_BASE_DIR=./bert/pretrain_model/chinese_L-12_H-768_A-12

nohup python -u ./bert/run_custom_classifier.py \
    --task_name='gt' \
    --do_lower_case=true \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --save_for_serving=true \
    --data_dir='../data' \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5\
    --num_train_epochs=1.0 \
    --use_gpu=true \
    --num_gpu_cores=4 \
    --use_fp16=false \
    --output_dir='./outputs' > train.log 2>&1 &
  
```

Shell script is available also (see shell_scripts/run_multi_gpu.sh)


* 根据`Dockerfile` 生成训练和预测的镜像，并且推送到`ECR`，注意这边需要切换到根路径
        
```
cd ./source
sh build_and_push.sh bert-sentiment-anylsis
```

* 用`source/bert/tensorflow_bring_your_own.ipynb`启动训练任务，并且生成模型文件保存在s3

此刻你可以看到你的SageMaker 控制台中生成了对应的`Training Job`


### Deploy
* 利用`EndpointDeploy.py`，使用`模型文件`和`Docker Image`和`.source/bert/run_classifier.py`生成`endpoint`

```sh
cd ./source
python EndpointDeploy.py \
--ecr_image_path="847380964353.dkr.ecr.us-east-1.amazonaws.com/bert-sentiment-anylsis:latest" \
--model_s3_path="s3://sagemaker-us-east-1-847380964353/model/model.tar.gz" \
--instance_type="ml.m4.xlarge"
```
此刻你可以看到你的SageMaker 控制台中生成了对应的`endpoint`


### Bot - 使用docker进行部署的机器人

机器人包含`Dockerfile`,`task.py`脚本，及相关依赖，目录结构如下

```
bot--|--dependency()--|--extract_features.py
     |                |--modeling.py
     |                |--tokenization.py
     |                |--vocab.txt
     |--Dockerfile
     |--task.py(执行主程序)
```

在任意ec2上运行如下命令即可build docker，运行对应的机器人任务
```sh
cd ./bot 
docker build -t ${DOCKER_IMAGE_NAME} .
docker run ${DOCKER_IMAGE_NAME}
```



