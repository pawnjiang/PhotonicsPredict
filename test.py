from os import X_OK

import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras import optimizers 
import numpy as np
# import os
import json
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


'''
    mnist image是28*28 shape,定义LSTM的input为(28, ),
    将image一行一行地输入到LSTM地cel中，这样time_step就是28，
    表示一个image有28行，LSTM地output是30个
'''


def computeR2(y_true, y_predict):
    y_true_mean = np.mean(np.array(y_true))
    u = 0
    v = 0
    for i in range(len(y_true)): # 实例个数
        u += (y_true[i] - y_predict[i])**2
        v += (y_true[i] - y_true_mean)**2
    return 1 - u/v

DATA_LENGTH = 5000
# DATA_LENGTH = 20
TOPK = 20
DATA_SIZE = 750
date = '2022.3.8'
filename = '20220308ppt'

# parameters for LSTM

nb_lstm_outputs = 1 # 神经元个数
# nb_time_step = 2 # 时间序列长度。我们的序列，这个值应该是1
nb_time_step = 1 # 时间序列长度。我们的序列，这个值应该是1
nb_input_vector = DATA_LENGTH # 输入序列.我们的序列，这个值应该是5000

# STEP 1 数据预处理
# label要用one_hot encoding, x_train的shape是(数据量，time_step, 序列长度)
concen_list = ['20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70']

def DataPro():
    global concen_list, date, filename
    dic_all = {}
    for concen in concen_list:
        file = open('data//' + filename + '//' + concen + '.txt', 'r')
        data_list = []
        for i in range(DATA_SIZE):
            one_seq = []
            for j in range(DATA_LENGTH*2):
                oneline = float(file.readline().strip())
                one_seq.append(oneline)
            data_list.append(one_seq[: DATA_LENGTH]) # 1w的数据前5k和后5k对称
        file.close()
        dic_all[concen] = data_list
    file = open('data//' + filename + '//'+ date +'_js' + '.txt','w')
    js = json.dumps(dic_all)
    file.write(js)
    file.close()

def test():
    global date, filename
    file = open('data//' + filename + '//'+ date +'_js' + '.txt','r')
    js = file.read()
    dic_all = json.loads(js)
    file.close()
    for key, value in dic_all.items():
        temp = np.array(value)
        print('type:{}'.format(type(temp)))
        print('key: {}, value shape: {}'.format(key, np.shape(temp)))

def GetData():
    global concen_list, date, filename
    # file = open('data//20220108//'+ date +'_js' + '.txt','r')
    # file = open('data//20220108//'+ '2022.1.8_20_onlyx_js.txt','r')
    # file = open('data//' + filename + '//'+ date +'_js' + '.txt','r')
    # file = open('data//20220108//'+ '2022.1.8' + '_' + str(TOPK) + '_onlyx_4.19_js' + '.txt','r')
    # file = open('data_final.csv','r')
    # js = file.read()
    # dic_all = json.loads(js)
    dic_all = pd.read_csv('data_final.csv',index_col=None)
    X = dic_all['data'].values
    y = dic_all[['lambda','power']].values
    Y=[]
    X = X.reshape((-1, DATA_LENGTH))
    for i in range(int(len(y)/5000)):
        Y.append(y[i*5000])
    print(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=True)

    # print('xt',x_train,'yt',y_train)
    # print(dic_all.)
    # print(dic_all['data'])
    # file.close()
    # x_train, y_train, x_test, y_test = [], [], [], []
    # for key, value in dic_all.items():
    #     print(key,value)
    #     # idx = concen_list.index(key)
    #     # idx = int(key)
    #     dist = int(DATA_SIZE*0.8)
    #     x_train += value[:dist,0]
    #     x_test += value[dist:,0]
    #     # y_train += [idx for i in range(dist)]
    #     # y_test += [idx for i in range(len(value) - dist)]
    #     y_train += value[:dist,0:]
    #     y_test += value[dist:,0:]
    # dist = int(DATA_SIZE * 0.8)
    # print(dic_all['data'])
    # x_train = dic_all['data'][:dist]
    # x_test = dic_all['data'][dist:]
    # y_train = dic_all[1:][:dist]
    # y_test = dic_all[1:][dist:]

    # randnum = random.randint(0,100)
    # random.seed(randnum)
    # random.shuffle(x_train)
    # random.seed(randnum)
    # random.shuffle(y_train)
    # randnum = random.randint(100,1000)
    # random.seed(randnum)
    # random.shuffle(x_test)
    # random.seed(randnum)
    # random.shuffle(y_test)
    # print(x_test,y_test)
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def GetDataTry():
    global concen_list, date, filename
    # file = open('data//20220108//'+ date +'_js' + '.txt','r')
    # file = open('data//20220108//'+ '2022.1.8_5000_js.txt','r')
    # file = open('data/' + filename + '//'+ date +'_js' + '.txt','r')
    # file = open('data//20220108//'+ '2022.1.8' + '_' + str(TOPK) + '_onlyx_4.19_js' + '.txt','r')
    file = open('data_final.csv','r')
    js = file.read()
    dic_all = json.loads(js)
    file.close()
    x_train, y_train, x_test, y_test = [], [], [], []
    dataset_num = 600
    TEST_NUM = 150
    i = 0
    for key, value in dic_all.items():
        if i%2==0:
            # idx = concen_list.index(key)
            x_train += value[:dataset_num]
            y_train += [int(key) for i in range(dataset_num)]
            # x_test += value[-1*TEST_NUM:]
            # y_test += [int(key) for i in range(TEST_NUM)]
            y_train += value[:dist, 0:]
            y_test += value[dist:, 0:]
        else:
            # x_test += value[-1*TEST_NUM:]
            # y_test += [int(key) for i in range(TEST_NUM)]
            x_test += value
            y_test += value[dist:, 0:]
        i += 1

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(x_train)
    random.seed(randnum)
    random.shuffle(y_train)
    randnum = random.randint(100,1000)
    random.seed(randnum)
    random.shuffle(x_test)
    random.seed(randnum)
    random.shuffle(y_test)
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
        
    


# DataPro() # 只执行一次即可
# test()

(x_train, y_train), (x_test, y_test) = GetData()

# (x_train, y_train), (x_test, y_test) = GetDataTry()
#
# lstm处理数据
# x_train = x_train.reshape((-1, nb_time_step, DATA_LENGTH))
# x_test = x_test.reshape((-1, nb_time_step, DATA_LENGTH))

# 正常处理数据
print('len_x:',len(x_train),'len_y:',len(y_train))
# print('shape:{}'.format(np.shape(y_train)))

x_train = x_train.reshape((-1, DATA_LENGTH))
# y_train = y_train.reshape((-1, 2, DATA_LENGTH))
# print(y_train)
x_test = x_test.reshape((-1, DATA_LENGTH))
# y_test = y_test.reshape((-1,2,DATA_LENGTH))
print('shape:{}'.format(np.shape(x_train)))
# print('shape:{}'.format(np.shape(y_train)))

print(len(y_train))

# y_train = to_categorical(y_train, num_classes=10) # to_categorical是对数字进行热编码
# y_test = to_categorical(y_test, num_classes=10) # to_categorical是对数字进行热编码
R2_list = []
for run_num in range(3,11,1):
    model = Sequential()
    # model.add(Dense(units=1000, input_dim=DATA_LENGTH, activation='relu'))
    # model.add(Dense(units=50, activation='relu'))
    # model.add(Dense(units=1, activation='relu'))
    model.add(Dense(units=100, input_dim=DATA_LENGTH, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation='relu'))

    # adam = Adam(0.0001)
    adam = Adam(0.0001)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1)
    model.fit(x_train, y_train, epochs=3000, batch_size=32, verbose=1)
    # model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=1)

    # STEP 5 evaluate
    # 可以使用model_summary()来查看神经网络的架构喝参数量等信息

    model.summary()

    score = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(score)

    y_predict = model.predict(x_test)


    plt.plot()
    for i in range(len(y_predict)):
        # print('y_predict:{}, y_actual:{}'.format(y_predict[i], y_test[i]))
        plt.plot(y_test[i], y_predict[i],c='red',alpha=0.05,marker='o'  )
    plt.plot([25,85],[25,85],c='blue',ls='--')
    plt.savefig(str(run_num)+'.png')
    plt.cla()
    # plt.show()

    R2 = computeR2(y_test, y_predict)
    R2_list.append(R2)
    # print('R2:{}'.format(R2))
    file = open('d4.22_bp_y_true'+ str(run_num)+ '.txt','w')
    for one in y_test:
        file.write(str(one)+'\n')
    file.close()

    file = open('d4.22_bp_y_predict'+ str(run_num)+ '.txt','w')
    for one in y_predict:
        file.write(str(one[0])+'\n')
    file.close()

    file = open('d4.22_bp_R2'+ str(run_num)+ '.txt','a+')
    file.write(str(R2)+'\n')
    file.close()

print('R2_list')
for i in range(R2_list):
    print(i)
