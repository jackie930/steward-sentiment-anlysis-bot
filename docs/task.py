#!/usr/bin/env python
# encoding: utf-8

import argparse
import os

import sys

sys.path.append("./dependency")
import json
import numpy as np
from boto3.session import Session
import tokenization
from extract_features import InputExample, convert_examples_to_features

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_s3_path_list", type=str)
parser.add_argument("-e", "--endpoint_name", type=str)
parser.add_argument("-o", "--output_s3_bucket", type=str)
parser.add_argument("-id", "--aws_access_key_id", type=str)
parser.add_argument("-secret", "--aws_secret_access_key", type=str)
parser.add_argument("-r", "--region_name", type=str)

# parser.add_argument("-e", "--es_host", type=str, help="http://2.3.4.5:8080")
# parser.add_argument("-n", "--batch_number", type=int, help="Use the number to get the shard for this bot.")
args = parser.parse_args()
# print("batch id is: {}, es url is: {}".format(args.batch_number, args.es_host))

# print("outside env- {}, batch_id - {}".format(os.getenv("ES_URL"), os.getenv("BATCH_ID")))

# print('Task starting...')
# time.sleep(3)
# print('Task ended, took 5 seconds')

def sentiment_analysis_main(
    input_s3_path_list,
    endpoint_name,
    output_s3_bucket,
    aws_access_key_id,
    aws_secret_access_key,
    region_name,
):
    """
 function: save json_files back to s3 (key: file name value: tag label)
    :param input_s3_path_list: 读入s3文件路径list
    :param endpoint_path: 保存名称
    """
    session = Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    s3 = session.client("s3")
    print("start!")
    input_s3_path_list = input_s3_path_list.split(",")
    for x in input_s3_path_list:
        result = {}
        bucket = x.split("/")[2]
        key = "/".join(x.split("/")[3:])
        file_name = x.split("/")[-1]
        # try:
        # s3.Bucket(bucket).download_file(key, file_name)
        print(file_name, key, bucket)
        s3.download_file(Filename=file_name, Key=key, Bucket=bucket)

        # read txt
        print("process", x)
        f = open(file_name, "r", encoding="utf-8")
        text = "".join(f.readlines())
        f.close()

        # infer endpoint
        label = single_bert_infer(session, endpoint_name, text)
        result[x] = label

        # save json file
        json_file = file_name.replace(".txt", ".json")
        with open(json_file, "w") as fw:  # 建议改为.split('.')
            json.dump(result, fw, ensure_ascii=False)
            print("write json file success!")

        # output to s3
        s3.upload_file(Filename=json_file, Key=json_file, Bucket=output_s3_bucket)

        # delete file locally
        delete_file(json_file)
        delete_file(file_name)


def delete_file(file):
    """
    delete file
    :param file:
    :return:
    """
    if os.path.isfile(file):
        try:
            os.remove(file)
        except:
            pass


def preprocess(text):
    """
 function: preprocess text into input numpy array
    """
    vocab_file = os.environ.get("vocab_file", "./dependency/vocab.txt")
    max_token_len = os.environ.get("max_token_len", 128)
    text_a = text
    example = InputExample(unique_id=None, text_a=text_a, text_b=None)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    feature = convert_examples_to_features([example], max_token_len, tokenizer)[0]
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return {"inputs": {"input_ids": input_ids.tolist()}}


def single_bert_infer(session, endpoint_name, text):
    """
 function: use endpoint to infer on one single text
    """
    # first preprocess input text
    print(endpoint_name)
    data = preprocess(text)
    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(data),
    )

    result = json.loads(response["Body"].read())
    pro_0, pro_1 = result["outputs"][0]
    # return tag negative/positive which have higher probability
    if pro_0 > pro_1:
        return "negative"
    else:
        return "positive"


if __name__ == "__main__":
    INPUT_S3_PATH_LIST = args.input_s3_path_list
    ENDPOINT_NAME = args.endpoint_name
    OUTPUT_S3_BUCKET = args.output_s3_bucket
    AWS_ACCESS_KEY_ID = args.aws_access_key_id
    AWS_SECRET_ASSESS_KEY = args.aws_secret_access_key
    REGION_NAME = args.region_name
    sentiment_analysis_main(
        INPUT_S3_PATH_LIST,
        ENDPOINT_NAME,
        OUTPUT_S3_BUCKET,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ASSESS_KEY,
        REGION_NAME,
    )
