# -*- coding: utf-8 -*-
# @Time    : 10/27/20 5:50 PM
# @Author  : Jackie
# @File    : preprocess_gt.py
# @Software: PyCharm

import os
import pandas as pd
from sklearn.model_selection import train_test_split

#process train
def process_data(input_file,label_col1,label_col2,text_col,save_path):
    """Pred folder contains info to be test on"""
    df = pd.read_csv(input_file)
    print (df.head())
    # save as defined
    df1 = df[[text_col,label_col1,label_col2]]
    df1.columns=['text_a','label_0','label_1']
    print (df1.head())

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_train,df_test = train_test_split(df1,test_size = 0.2)
    df_train.to_csv(os.path.join(save_path,'train.tsv'),sep='\t',index=False)
    df_test.to_csv(os.path.join(save_path,'test.tsv'),sep='\t',index=False)
    df_test.to_csv(os.path.join(save_path,'dev.tsv'),sep='\t',index=False)

    print ("<<<< save train,test,dev result")


if __name__ == "__main__":
    #exp4 fore, enhanced
    print ("<<<<Start")
    input_file = '/Users/liujunyi/Desktop/spottag/tcl/data.csv'
    save_path = './res'
    label_col1 = 'x5'
    label_col2 = 'x6'
    text_col = 'text'
    process_data(input_file,label_col1,label_col2,text_col,save_path)
