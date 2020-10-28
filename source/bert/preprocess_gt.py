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
    df=ls[0]
    for i in range(1,len(ls)):
        df=df.append(ls[i])
    df.to_csv(os.path.join(save_path,'org_pred.csv'),index=False)
    # save as defined
    df1 = df[['label','input']]
    df1.columns=['label','text_a']
    df1.to_csv(os.path.join(save_path,'dev.tsv'),sep='\t',index=False)
    df1.to_csv(os.path.join(save_path,'test.tsv'),sep='\t',index=False)
    print ("<<<< save test result")
    print ("<<<<<<df shape",df1.shape)


#process train
def process_data_train(input_train_file,save_path):
    """Pred folder contains info to be test on"""
    df = pd.read_csv(input_train_file)
    # save as defined
    df1 = df[['label','text_a']]
    print (df1.label.unique())
    df1.columns=['label','text_a']
    #count the distribution
    gp=df.groupby(by=['label'])
    print ("<<<< TRAIN LABEL Distribution", gp.size())
    df1.to_csv(os.path.join(save_path,'train.tsv'),sep='\t',index=False)
    print ("<<<< save train result")
    print ("<<<<<<df shape",df1.shape)

def main(pred_folder,input_train_file,save_path):
    process_data_pred(pred_folder,save_path)
    process_data_train(input_train_file,save_path)

if __name__ == "__main__":
    #exp1, hind, imbalanced
    print ("<<<<EXP 1")
    pred_folder = '../gt_data/aws_results_v2/hind'
    input_train_file = '../gt_data/exp_v2/exp_h_v2.csv'
    save_path =  '../experiments/exp1'
    main(pred_folder,input_train_file,save_path)

    #exp2, hind, enhanced
    print ("<<<<EXP 2")
    pred_folder = '../gt_data/aws_results_v2/hind'
    input_train_file = '../gt_data/exp_v2/exp_h_enhance_v2.csv'
    save_path =  '../experiments/exp2'
    main(pred_folder,input_train_file,save_path)

    #exp3 fore, imbalanced
    print ("<<<<EXP 3")
    pred_folder = '../gt_data/aws_results_v2/fore'
    input_train_file = '../gt_data/exp_v2/exp_f_v2.csv'
    save_path =  '../experiments/exp3'
    main(pred_folder,input_train_file,save_path)

    #exp4 fore, enhanced
    print ("<<<<EXP 4")
    pred_folder = '../gt_data/aws_results_v2/fore'
    input_train_file = '../gt_data/exp_v2/exp_f_enhance_v2.csv'
    save_path =  '../experiments/exp4'
    main(pred_folder,input_train_file,save_path)
