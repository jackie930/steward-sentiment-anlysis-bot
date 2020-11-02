# -*- coding: utf-8 -*-
# @Time    : 10/27/20 5:50 PM
# @Author  : Jackie
# @File    : preprocess_gt.py
# @Software: PyCharm

import os
import pandas as pd
from sklearn.model_selection import train_test_split

#process train
def process_data(input_file,label_col,text_col,save_path):
    """Pred folder contains info to be test on"""
    df = pd.read_csv(input_file)
    print (df.head())
    # save as defined
    df1 = df[[label_col,text_col]]
    print (df1[label_col].unique())
    df1.columns=['label','text_a']
    print (df1.head())
    #count the distribution
    gp=df1.groupby(by=['label'])
    print ("<<<< TRAIN LABEL Distribution", gp.size())

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_train,df_test = train_test_split(df1,test_size = 0.2,stratify=df1['label'])
    df_train.to_csv(os.path.join(save_path,'train.tsv'),sep='\t',index=False)
    df_test.to_csv(os.path.join(save_path,'test.tsv'),sep='\t',index=False)
    df_test.to_csv(os.path.join(save_path,'dev.tsv'),sep='\t',index=False)

    print ("<<<< save train,test,dev result")


if __name__ == "__main__":
    #exp4 fore, enhanced
    print ("<<<<EXP 4")
    input_file = '/Users/liujunyi/Desktop/spottag/tcl/data.csv'
    save_path = './res'
    label_col = 'x6'
    text_col = 'text'
    process_data(input_file,label_col,text_col,save_path)
