# -*- coding: utf-8 -*-
# @Time    : 10/27/20 5:50 PM
# @Author  : Jackie
# @File    : preprocess_gt.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def main(test_output_path,test_org_path,save_path):
    res = pd.read_csv('./outputs/exp2/test_results.tsv',sep='\t',header=None)
    res.columns=['1','0','-1']
    data = res.values

    res['max_value']=res.max(axis=1)
    res['max_index']=np.argmax(data,axis=1)
    ls = [1,0,-1]
    res.loc[(res.max_index == 0) ,'pred_label'] = 1
    res.loc[(res.max_index == 1) ,'pred_label'] = 0
    res.loc[(res.max_index == 2) ,'pred_label'] = -1

    test = pd.read_csv(test_org_path)
    test=test.drop(["pred_label"],axis=1)
    res2= pd.concat([test,res], axis=1)
    #save results
    res2.to_csv(save_path)

    #print accuracy
    y_true=res2.label
    y_pred=res2.pred_label


    f1 = f1_score( y_true, y_pred, average='macro' )
    print ("macro F1",f1)

    p = precision_score(y_true, y_pred, average='macro')
    print ("macro precision_score",p)

    r = recall_score(y_true, y_pred, average='macro')
    print ("macro recall_score",r)

if __name__ == "__main__":
    test_output_path='../outputs/test_results.tsv'
    test_org_path='../../experiments/exp4/org_pred.csv'
    save_path='../../experiments/exp4/res.csv'
    main(test_output_path,test_org_path,save_path)

