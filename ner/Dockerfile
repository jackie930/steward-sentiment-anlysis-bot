# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.5-gpu-py36-cu100-ubuntu18.04
# FROM 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-training:1.15.5-gpu-py36-cu100-ubuntu18.04
FROM tensorflow/tensorflow:1.15.2-gpu-py3
#FROM 343958593302.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-tf-albert-chinese-ner:latest

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

# Download TensorFlow Serving
# https://www.tensorflow.org/serving/setup#installing_the_modelserver
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY ./ /opt/ml/code
WORKDIR /opt/ml/code