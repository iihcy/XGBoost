#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


def featureSet(data):
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    yList = data.y.values
    hh = np.mat(XList)  # 10441*19
    gg = yList  # 10441*1
    return XList, yList


def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    # data = np.mat(data)


    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer.fit(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])
    x_new = imputer.transform(data.loc[:, ['rw', 'st', 'lw', 'cf', 'cam', 'cm']])

    le = preprocessing.LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    att_label = le.transform(data.work_rate_att.values)
    # print(att_label)
    def_label = le.transform(data.work_rate_def.values)
    # print(def_label)

    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['club'])
        tmp_list.append(data.iloc[row]['league'])
        tmp_list.append(data.iloc[row]['potential'])
        tmp_list.append(data.iloc[row]['international_reputation'])
        tmp_list.append(data.iloc[row]['pac'])
        tmp_list.append(data.iloc[row]['sho'])
        tmp_list.append(data.iloc[row]['pas'])
        tmp_list.append(data.iloc[row]['dri'])
        tmp_list.append(data.iloc[row]['def'])
        tmp_list.append(data.iloc[row]['phy'])
        tmp_list.append(data.iloc[row]['skill_moves'])
        tmp_list.append(x_new[row][0])
        tmp_list.append(x_new[row][1])
        tmp_list.append(x_new[row][2])
        tmp_list.append(x_new[row][3])
        tmp_list.append(x_new[row][4])
        tmp_list.append(x_new[row][5])
        tmp_list.append(att_label[row])
        tmp_list.append(def_label[row])
        XList.append(tmp_list)
    return XList


def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=500, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(10441, 17441)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)

    # 显示重要特征
    plot_importance(model)
    print('===============================================')
    feature_importances_ = model.feature_importances_
    print(feature_importances_)
    print(np.shape(feature_importances_))
    print('===============================================')
    plt.show()

if __name__ == '__main__':
    # 文件读取
    trainFilePath = 'train.csv'
    testFilePath = 'test.csv'
    data = pd.read_csv(trainFilePath)
    # 训练数据准备：X_train = 10441*19；y_train = 10441*1
    X_train, y_train = featureSet(data)
    # X_train = np.mat(X_train)
    # y_train = np.mat(y_train)
    # m = np.shape(X_train)
    # n_x = np.shape(y_train)
    # 测试数据准备：X_test = 7000*19
    X_test = loadTestData(testFilePath)
    # X_test = np.mat(X_test)
    # n_y = np.shape(X_test)
    # 模型建立
    trainandTest(X_train, y_train, X_test)
