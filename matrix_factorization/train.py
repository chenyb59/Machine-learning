from pmf_batchtrain_v import *
import numpy as np
import matplotlib.pyplot as plt
import math


max_r = 5
min_r = 1

user_index = {}
item_index = {}
max_user_index = 0
max_item_index = 0


## map the original id into integer
def get_index(u_str , i_str):
    global max_user_index , max_item_index
    if user_index == {}:
        user_index[u_str] = 0
    if user_index.get(u_str) == None:
        user_index[u_str] = max_user_index + 1
        max_user_index = max_user_index + 1

    if item_index == {}:
        item_index[i_str] = 0
    if item_index.get(i_str) == None:
        item_index[i_str] = max_item_index + 1
        max_item_index = max_item_index + 1

    return user_index.get(u_str) , item_index.get(i_str)


## detect the user and item id to find whether in the train data
def adjust(set_train , set_test):
    u_list = np.unique(set_train[:,0])
    i_list = np.unique(set_train[:,1])
    user_not_list = []
    item_not_list = []
    in_list = []
    all_not_list = []
    for f in range(set_test.shape[0]):
        print f
        if np.isin(set_test[f,0] , u_list) & np.isin(set_test[f,1] , i_list):
            in_list.append(f)
        else:
            if np.isin(set_test[f,1] , i_list):
                user_not_list.append([f,set_test[f,1]])
            elif np.isin(set_test[f,0] , u_list):
                item_not_list.append([f,set_test[f,0]])
            else:
                print 'both user and item not found ! row : ',f
                all_not_list.append(f)
    return in_list,all_not_list,user_not_list,item_not_list

## adjust the predict rating : not larger than 5, not smaller than 1
def normal_rate(rl):
    rl[np.where(rl<1)] = 1.00
    rl[np.where(rl>5)] = 5.00
    return rl


def main():
    train_path = 'train.dat'
    test_path = 'test.dat'

## read train data and test data from files
    data = []
    f = open(train_path)
    line = f.readline()
    while line:
        line = line.strip()
        info = line.split('\t')
        i , j= get_index(info[0] , info[1])
        data.append([i , j ,float(info[2])])
        line = f.readline()
    f.close()
    train_data = np.array(data , dtype = 'int32')

    data = []
    f = open(test_path)
    line = f.readline()
    while line:
        line = line.strip()
        info = line.split('\t')
        i , j= get_index(info[0] , info[1])
        data.append([i , j ])
        line = f.readline()
    f.close()
    test_data = np.array(data , dtype = 'int32')


## adjust the test data : seperate the data which contain unseen id in training
## recommendation policy :
## 1. if both the user id and item id are contained in train data , predict the rating by user_feature * item_feature;
## 2. if the user id is not contained in train data , predict the rating by calculating the mean of history ratings of the item;
## 3. if the item id is not contained in train data , predict the rating by calculating the mean of history ratings of the user;
## 4. if both the user id and item id are not contained in train data , predict the rating by the most frequent rating of train data.

    in_list ,all_not_list, user_n_list,item_n_list = adjust(train_data , test_data)


    dim = 10
    lamda_u = 0.1
    lamda_v = 0.1
    epoch_num = 20
    batch_size = 32
    learning_rate = 2.5

    user_feature , item_feature = pmf_batch(train_data , dim , learning_rate ,lamda_u , lamda_v , epoch_num , batch_size)


    for_predict = test_data[in_list , :]
    raw_r = np.sum(np.multiply(user_feature[:,for_predict[:,0]] , item_feature[: , for_predict[:,1]]) , axis = 0).reshape(-1,1)
    predict_r = normal_rate(raw_r)
    in_list = np.array(in_list , dtype = 'int32').reshape(-1,1)
    total_r = np.concatenate((in_list , predict_r) , axis =1)


    for up in user_n_list:
        m1 = np.mean(train_data[train_data[:,1]==up[1] , 2])
        total_r = np.append(total_r , [[up[0] , m1]] , axis = 0)

    for ip in item_n_list:
        m1 = np.mean(train_data[train_data[:,0]==ip[1] , 2])
        total_r = np.append(total_r , [[ip[0] , m1]] , axis = 0)


    counts = np.bincount(train_data[:,2])
    m2 = np.argmax(counts)
    for p in all_not_list:
        total_r = np.append(total_r , [[p , m2]] , axis = 0)


    sorted_r = total_r[total_r[:, 0].argsort()]
    fw = open('test_output.txt','w')
    for rating in sorted_r:
        fw.write(str(float('%.2f' % rating[1])))
        fw.write('\n')
    fw.close()



if __name__ == "__main__":
    main()
