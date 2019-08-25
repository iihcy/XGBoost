#coding:utf-8
# _author_=hcy

'''
    模型解说：利用XGBoost模型选出重要特征；
    其次对全局的特征进行得分（节点分裂后平方损失的减少值）排序，并进行截取K个特征；
    最后将XGBoost选出的特征子集（需迭代确定个数）代替PLS的主成分，进行回归分析；
'''
from numpy import *
from sklearn import preprocessing
import random
import xgboost as xgb
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
import re
import xlwt
# import Data_matching
import time

#数据读取-单因变量与多因变量
def loadDataSet01(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    row = len(arrayLines)
    x = mat(zeros((row, 9)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index, :] = curLine[0:9]
        y[index, :] = curLine[-1]
        index += 1
    return x, y

'''
为了获得较可靠的结果，需测试数据（测验）和训练数据（学习）
--可按照比例划分数据集,即随机分成训练集与测试集
'''
def splitDataSet(x, y):
    m =shape(x)[0]
    train_sum = int(round(m * 0.7))
    test_sum = m - train_sum
    #利用range()获得样本序列
    randomData = range(0, m)
    randomData = list(randomData)
    #根据样本序列进行分割- random.sample(A,rep)
    train_List = random.sample(randomData, train_sum)
    #获取训练集数据-train
    train_x = x[train_List, :]
    train_y = y[train_List, :]
    #获取测试集数据-test
    test_list = []
    for i in randomData:
        if i in train_List:
            continue
        test_list.append(i)
    test_x = x[test_list,:]
    test_y = y[test_list,:]
    return train_x, train_y, test_x, test_y

#数据标准化
def stardantDataSet(x0, y0):
    e0 = preprocessing.scale(x0)
    f0 = preprocessing.scale(y0)
    return e0, f0

#求均值-标准差
def data_Mean_Std(x0, y0):
    mean_x = mean(x0, 0)
    mean_y = mean(y0, 0)
    std_x = std(x0, axis=0, ddof=1)
    std_y = std(y0, axis=0, ddof=1)
    return mean_x, mean_y, std_x, std_y

#PLS核心函数
def PLS(x0, y0):
    e0, f0 = stardantDataSet(x0, y0)
    e0 = mat(e0); f0 = mat(f0); m = shape(x0)[1]; ny=shape(y0)[1]
    w = mat(zeros((m, m))).T; w_star = mat(zeros((m, m))).T
    chg = mat(eye((m)))
    my = shape(x0)[0];ss = mat(zeros((m, 1))).T
    t = mat(zeros((my,m))); alpha= mat(zeros((m, m)))
    press_i = mat(zeros((1,my)))
    press = mat(zeros((1, m)))
    Q_h2 = mat(zeros((1, m)))
    beta = mat(zeros((1,m))).T
    for i in range(1,m+1):
        #计算w,w*和t的得分向量
        matrix = e0.T * f0 * (f0.T * e0)
        val, vec = linalg.eig(matrix)#求特征向量和特征值
        sort_val = argsort(val)
        index_vec = sort_val[:-2:-1]
        w[:,i-1] = vec[:,index_vec]#求最大特征值对应的特征向量
        w_star[:,i-1] =  chg * w[:,i-1]
        t[:,i-1] = e0 * w[:,i-1]
        #temp_t[:,i-1] = t[:,i-1]
        alpha[:,i-1] = (e0.T * t[:,i-1]) / (t[:,i-1].T * t[:,i-1])
        chg = chg * mat(eye((m)) - w[:,i-1] * alpha[:,i-1].T)
        e = e0 - t[:,i-1] * alpha[:,i-1].T
        e0 = e
        #计算ss(i)的值
        #beta = linalg.inv(t[:,1:i-1], ones((my, 1))) * f0
        #temp_t = hstack((t[:,i-1], ones((my,1))))
        #beta = f0\linalg.inv(temp_t)
        #beta = nnls(temp_t, f0)
        beta[i-1,:] = (t[:,i-1].T * f0) /(t[:,i-1].T * t[:,i-1])
        cancha = f0 - t * beta
        ss[:,i-1] = sum(sum(power(cancha, 2),0),1)#注：对不对？？？
        for j in range(1,my+1):
            if i==1:
                t1 = t[:, i - 1]
            else:
                t1 = t[:,0:i]
            f1=f0
            she_t = t1[j-1,:]; she_f = f1[j-1,:]
            t1=list(t1); f1 = list(f1)
            del t1[j-1];  del f1[j-1] #删除第j-1个观察值
            #t11 = np.matrix(t1)
            #f11 = np.matrix(f1)
            t1 = array(t1); f1 = array(f1)
            if i==1:
                t1 = mat(t1).T; f1 = mat(f1).T
            else:
                t1 = mat(t1); f1 = mat(f1).T

            beta1 = linalg.inv(t1.T * t1) * (t1.T * f1)
            #beta1 = (t1.T * f1) /(t1.T * t1)#error？？？
            cancha = she_f - she_t*beta1
            press_i[:, j-1] = sum(power(cancha,2))
        press[:, i-1]=sum(press_i)
        if i>1:
            Q_h2[:, i-1] =1-press[:,i-1]/ss[:,i-2]
        else:
            Q_h2[:, 0]=1
        if Q_h2[:, i-1] < 0.0975:
            h = i
            break
    return h, w_star, t, beta

##计算反标准化之后的系数
def Calxishu(xishu, mean_x, mean_y, std_x, std_y):
    n = shape(mean_x)[1]; n1 = shape(mean_y)[1]
    xish = mat(zeros((n, n1)))
    ch0 = mat(zeros((1, n1)))
    for i in range(n1):
        ch0[:, i] = mean_y[:, i] - std_y[:, i] * mean_x / std_x * xishu[:, i]
        xish[:, i] = std_y[0, i] * xishu[:, i] / std_x.T
    return ch0, xish


# XGBoost模型建立：需要调参
def XGBoostModel(x0, y0):
    # XGBoost训练
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.006, n_estimators=500, silent=False, objective='reg:gamma')
    model.fit(x0, y0)
    # 对实验数据进行预测
    # xgy_predict = model.predict(x0)
    feature_Score = model.feature_importances_
    # fScore = pd.DataFrame(feature_Score)
    # score = model.base_score
    # print(feature_Score)
    # 画图显示
    # plot_importance(model)
    # plt.show()
    return feature_Score, model


# 选取前k个特征
def featureSelect(feature_Score, model, x0, y0, k, original_Data):
    row_y = shape(y0)[0]
    feature_Score = list(feature_Score)
    # 排序--reverse = True 降序，返回索引序列
    # feature_Score.sort(reverse=True)
    index_Sorts = sorted(range(len(feature_Score)), key=feature_Score.__getitem__, reverse=True)
    m = len(index_Sorts)
    Indexss = mat(zeros((m, 1)))
    for j in range(m):
        index_Sorts = mat(index_Sorts)
        Indexss[j,: ] = index_Sorts[:, j]
    # 选取前k个特征
    Indexss_Select = Indexss[0: k]
    n = len(Indexss_Select)
    # name_pipei = mat(zeros((n, 1)))
    name_pipei = []
    for i in range(n):
        Name_Select = Indexss_Select[i, :]
        Name_Select = str(Name_Select)
        row_name = len(Name_Select)
        # 匹配列名
        if (row_name == 6): # x1~x9
            Name_Select = Name_Select[2:3]
            x_name = str('x' + Name_Select)
        if (row_name == 7): # x10~x99
            Name_Select = Name_Select[2:4]
            x_name = str('x' + Name_Select)
        if (row_name == 8): # x100~x999
            Name_Select = Name_Select[2:5]
            x_name = str('x' + Name_Select)
        name_pipei.append(x_name)
    # 拼接数据与匹配数据
    name_pipei = mat(name_pipei).T
    mat_str = array('mat')
    mat_top = mat(mat_str)
    # mat_str = (mat_str)
    all_mat = vstack((mat_str, name_pipei))
    # 选出的特征名——文件存储
    file = xlwt.Workbook()
    sheet1 = file.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    sel_row = len(all_mat)
    for i in range(sel_row):
        sheet1.write(i, 0, re.sub('[\[\]\']+', '', str(all_mat[i])))
    file.save('mat_xgbResult.xls')

    # 数据选择：与原始数据进行匹配，选出子集
    select_x0 = data_Matching(original_Data)
    # 求特征选择后的XGBoost精度，遍历求最佳特征子集
    model.fit(select_x0, y0)
    xgb_ypredict = mat(model.predict(select_x0)).T  # x0为特征选取的矩阵
    # SSE = sum(sum(power((y0 - y_predict), 2), 0))
    # XGB_SSE = sum(sum(power((y0 - xgb_ypredict), 2), 0))
    XGB_SSE = sum(power((y0 - xgb_ypredict), 2), 0)
    XGB_RMSE = sqrt(XGB_SSE / row_y) #均方根误差
    return XGB_RMSE, select_x0


# 特征个数从1-m（特征共m个）进行遍历--寻优：求解最佳k值
def Iter_featureSelect(feature_Score, model, x0, y0, original_Data):
    m = shape(x0)[1]
    ALLXGB_RMSE = []
    MINtemp_RMSE = 0
    feature_Numbel = 0
    for k in range(1, m+1):
        XGB_RMSE, select_x0 = featureSelect(feature_Score, model, x0, y0, k, original_Data)
        ALLXGB_RMSE.append(XGB_RMSE)
    ALLXGB_RMSE = mat(array(ALLXGB_RMSE)).T
    # 根据最佳的RMSE（XGBoost）寻找最佳的特征子集
    for j in range(0, m-1):
        temp_RMSE = ALLXGB_RMSE[j, :]
        if (temp_RMSE > ALLXGB_RMSE[j+1, :]):
            MINtemp_RMSE = ALLXGB_RMSE[j + 1, :] # 减去一次，如果全选就无特征选择的意义
            feature_Numbel = j + 2 # 特征个数索引,即k值
    MINtemp_RMSE = mat(MINtemp_RMSE)
    select_x0 = mat(select_x0)
    return ALLXGB_RMSE, MINtemp_RMSE, feature_Numbel, select_x0

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
    selectData = mat(selectData)
    n = selectData.shape[0]
    print(n)
    allColName = array(allcolName)
    selectData = mat(selectData)
    AllselectDatas = []
    for i in range(m):
        allNameI = allcolName[i]
        for j in range(n):
            f_s = selectData[j, :]
            f_str = str(f_s)
            s_name = re.sub('[[\\]]', '', f_str)
            sJ_Name = re.sub('\'', '', s_name)
            if allNameI == sJ_Name:
                allData = mat(allData)
                ff= allData[:, i]
                AllselectDatas.append(ff)
    AllselectDatas = array(AllselectDatas)
    AllselectDatas = mat(AllselectDatas).T
    return AllselectDatas


# 主函数
if __name__ == '__main__':
    starttime = time.clock()
    original_Data = pd.read_csv("TCMdata.csv")
    x0, y0 = loadDataSet01('TCMdata.txt')#单因变量与多因变量
    # 随机划分数据集
    train_x, train_y, test_x, test_y = splitDataSet(x0, y0)

    # XGBoost建模
    feature_Score, model = XGBoostModel(x0, y0)
    ALLXGB_RMSE, MINtemp_RMSE, k, select_x0 = Iter_featureSelect(feature_Score, model, x0, y0, original_Data)
    XGB_RMSE, select_x0 = featureSelect(feature_Score, model, x0, y0, k, original_Data)
    ALLXGB_RMSE = mat(ALLXGB_RMSE).T

    # 标准化
    e0, f0 = stardantDataSet(select_x0, y0)
    mean_x, mean_y, std_x, std_y = data_Mean_Std(select_x0, y0)
    r = corrcoef(select_x0)
    m = shape(select_x0)[1]
    n = shape(y0)[1]  # 自变量和因变量个数
    row = shape(select_x0)[0]
    #PLS函数
    h, w_star, t, beta = PLS(select_x0, y0)
    xishu = w_star * beta
    #反标准化
    ch0, xish = Calxishu(xishu, mean_x, mean_y, std_x, std_y)
    # 求可决系数和均方根误差
    y_predict = select_x0 * xish + tile(ch0[0, :], (row, 1))
    y_mean = tile(mean_y, (row, 1))
    SSE = sum(sum(power((y0 - y_predict), 2), 0))
    SST = sum(sum(power((y0 - y_mean), 2), 0))
    SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    RR1 = SSR / SST
    RMSE = sqrt(SSE / row)
    endtime = time.clock()
    print("=============================")
    # print(u"每个特征的得分(已处理)：")
    # print(feature_Score)
    print(u"选取前k(从1-m)特征所对应的RMSE值:")
    print(ALLXGB_RMSE)
    print (u"最优选取的第", k, u"特征所对应的RMSE值:", MINtemp_RMSE)
    print('SSE:', SSE)
    print('SST:', SST)
    print('SSR:', SSR)
    print(u"R2:", RR1)
    print(u"RMSE:", RMSE)
    print("=============================")
    time_s = (endtime - starttime)
    print(u"run_time: %fs" % time_s)