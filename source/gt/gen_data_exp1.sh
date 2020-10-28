#!/usr/bin/env bash
# execute the task
echo "task starts..."

#download data from s3
mkdir ../gt_data
aws s3 cp s3://cathysite-ecv ../gt_data --recursive
unzip ../gt_data/*.zip

#run preprocess
python ../bert/preprocess_gt.py

echo "task is done..."

