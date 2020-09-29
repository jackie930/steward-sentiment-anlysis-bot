# steward-sentiment-anlysis-bot
基于Google-Bert,进行Sentiment Analysis的任务，并利用`AWS SageMaker`进行模型训练和部署。

## Data
本解决方案使用的数据分为两部分，预训练模型的数据和
* 使用的基础模型是从google发布的bert预训练模型得到的，[模型下载地址](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
* 使用的情感分析数据集是新浪微博的短文本，有10万条评论数据，公开数据已经标注了正负向的情感标注

## Model
模型是end-to-end的二分类模型，[模型论文](https://arxiv.org/abs/1810.04805)

2018年google推出了bert模型，这个模型的性能要远超于以前所使用的模型，总的来说就是很牛。但是训练bert模型是异常昂贵的，对于一般人来说并不需要自己单独训练bert，只需要加载预训练模型，就可以完成相应的任务。

## Quick Start Guide
### Train
使用`SageMaker BYOC`训练的步骤
* 下载预训练模型，放到`./source/bert/pretrain_model`目录下，模型大小364.20M

```
wget -P ./source/bert/pretrain_model https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
cd ./source/bert/pretrain_model
unzip chinese_L-12_H-768_A-12.zip 
```

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



