#!/bin/bash
# set -x
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
profile=$2

if [ "$image" == "" ]
then
    echo "Use image name autogluon-sagemaker-inference-bot"
    image="autogluon-sagemaker-inference-bot"
fi

if [ "$profile" == "" ]
then
    echo "Use profile=default"
    profile="default"
fi

chmod +x bert/train
chmod +x bert/serve

# Get the account number associated with the current IAM credentials
account=$(aws --profile ${profile} sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration
# region=$(aws --profile ${profile} configure get region)
regions="us-east-1 us-east-2"

for region in $regions; do

if [[ $region =~ ^cn.* ]]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:latest"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
fi

echo ${fullname}

# If the repository doesn't exist in ECR, create it.
aws --profile ${profile} ecr describe-repositories --repository-names "${image}" --region ${region}

if [ $? -ne 0 ]
then
    aws --profile ${profile} ecr create-repository --repository-name "${image}" --region ${region} > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws --profile ${profile} ecr get-login --registry-ids ${account} --region ${region} --no-include-email)

# Build the docker image, tag with full name and then push it to ECR
docker build -t ${image} -f Dockerfile .
docker tag ${image} ${fullname}
docker push ${fullname}

done
