# -*- coding: utf-8 -*-
# @Time    : 10/27/20 5:50 PM
# @Author  : Jackie
# @File    : preprocess_gt.py
# @Software: PyCharm

import os
import pandas as pd

#process pred
def process_data_pred(pred_folder,save_path):
    """Pred folder contains info to be test on"""
    files = os.listdir(pred_folder)
    ls = []
    for i in files:
        if i.split('.')[-1]=='csv':
            print (i)
            data = pd.read_csv(os.path.join(pred_folder,i))
            ls.append(data)
    print(ls)
    df=ls[0]
    for i in range(1,len(ls)):
        df=df.append(ls[i])
    print ("<<<<<<df shape", df.shape)
    print (df.columns)
    df.to_csv(os.path.join(save_path,'org_pred.csv'),index=False)
    # save as defined
    df1 = df[['label','input']]
    df1.columns=['label','text_a']
    df1.to_csv(os.path.join(save_path,'dev.tsv'),sep='\t',index=False)
    df1.to_csv(os.path.join(save_path,'test.tsv'),sep='\t',index=False)
    print ("<<<< save test result")
    print (df1.head())


#process train
def process_data_train(input_train_file,save_path):
    """Pred folder contains info to be test on"""
    df = pd.read_csv(input_train_file)
    print ("<<<<<<df shape", df.shape)
    print (df.columns)
    # save as defined
    df1 = df[['label','text_a']]
    print (df1.label.unique())
    df1.columns=['label','text_a']
    df1.to_csv(os.path.join(save_path,'train.tsv'),sep='\t',index=False)
    print ("<<<< save train result")
    print (df1.head())

def main(pred_folder,input_train_file,save_path):
    process_data_pred(pred_folder,save_path)
    process_data_train(input_train_file,save_path)

if __name__ == "__main__":
    #exp1
    pred_folder = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/aws_results/hind'
    input_train_file = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/exp_h.csv'
    save_path =  '../output'
    main(pred_folder,input_train_file,save_path)

    #exp2
    pred_folder = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/aws_results/hind'
    input_train_file = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/exp_h.csv'
    save_path =  '../output'
    main(pred_folder,input_train_file,save_path)

    #exp3
    pred_folder = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/aws_results/hind'
    input_train_file = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/exp_h.csv'
    save_path =  '../output'
    main(pred_folder,input_train_file,save_path)

    #exp4
    pred_folder = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/aws_results/hind'
    input_train_file = '/Users/liujunyi/Desktop/spottag/cathysite/data-2/exp_h.csv'
    save_path =  '../output'
    main(pred_folder,input_train_file,save_path)

