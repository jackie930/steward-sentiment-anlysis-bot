# -*- coding: utf-8 -*-
# @Time    : 6/19/20 3:28 PM
# @Author  : Jackie
# @File    : EndpointDeploy.py
# @Software: PyCharm

import argparse
from sagemaker import get_execution_role
from sagemaker.tensorflow.model import TensorFlowModel

def init_args():
    """
    参数初始化
    :return: None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ecr_image_path", type=str, help="ecr image path",required=True
    )
    parser.add_argument(
        "--model_s3_path", type=str, help="s3 model file path",required=True
    )
    parser.add_argument(
        "--instance_type", type=str, help="deploy instance type",default='ml.m4.xlarge'
    )
    parser.add_argument(
        "--endpoint_name", type=str, help="deploy endpoint name",default='bert-sentiment-analysis'
    )

    return parser.parse_args()

def main(ecr_image_path,model_s3_path,instance_type,endpoint_name):
    role = get_execution_role()
    sagemaker_model = TensorFlowModel(model_data = model_s3_path,
                                      role = role,
                                      image = ecr_image_path,
                                      entry_point = 'bert/run_classifier.py'
                                      )
    predictor = sagemaker_model.deploy(initial_instance_count = 1,
                                       instance_type = instance_type,
                                       endpoint_name = endpoint_name)
    # get the status of the endpoint
    sagemaker = boto3.client(service_name='sagemaker')
    response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print('EndpointStatus = {}'.format(status))

    if status=='InService':
        print('Endpoint creation ended with EndpointStatus = {}'.format(status))
    else:
        print('Endpoint creation ended with EndpointStatus = {}'.format(status))

    return endpoint_name

if __name__ == "__main__":
    # init args
    ARGS = init_args()
    main(
        ecr_image_path=ARGS.ecr_image_path,
        model_s3_path=ARGS.model_s3_path,
        instance_type=ARGS.instance_type,
        endpoint_name=ARGS.endpoint_name
    )
