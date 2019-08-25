#coding:utf-8
# _author_=@hcy_Aliked

import numpy as np
import pandas as pd
import re
'''
    XGBoost算法特征排序后，选取前k个特征所对应的数据匹配
'''

def data_Matching(original_Data):
    allData = original_Data
    ##获取所有的列名
    allcolName = allData.columns.values.tolist()
    m = allData.shape[1]
    mm = allData.shape[0]
    print(m, mm)
    selectData = pd.read_excel("mat_xgbResult.xls")
    selectData = np.mat(selectData)
    n = selectData.shape[0]
    print(n)
    allColName = np.array(allcolName)
    selectData = np.mat(selectData)
    AllselectDatas = []
    for i in range(m):
        allNameI = allcolName[i]
        for j in range(n):
            f_s = selectData[j, :]
            f_str = str(f_s)
            s_name = re.sub('[[\\]]', '', f_str)
            sJ_Name = re.sub('\'', '', s_name)
            if allNameI == sJ_Name:
                allData = np.mat(allData)
                ff= allData[:, i]
                AllselectDatas.append(ff)

    AllselectDatas = np.array(AllselectDatas)
    AllselectDatas = np.mat(AllselectDatas).T
    # AllselectDatas = np.array(AllselectDatas)
    # selectData = np.array(selectData)
    # selectData = np.mat(selectData).T
    # selectData = np.array(selectData)
    # AselData = np.vstack((selectData, AllselectDatas))
    #
    # file = pd.DataFrame(AselData)
    # file.to_csv(u'#特征选择后的数据.csv')
    return AllselectDatas