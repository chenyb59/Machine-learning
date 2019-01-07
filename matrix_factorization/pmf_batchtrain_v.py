#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from math import *
import datetime
import collections


sigma_u = 0.01
sigma_v = 0.01


def compute_loss(type , rating , u , v , regu_u = 0 , regu_v = 0):
    if (np.amax(rating[: , 0]) > u.shape[1]) | (np.amax(rating[: , 1]) > v.shape[1]):
        print 'there is user id or item id exceed the trainset max id.'
    else:
        if type == 'train':
            p1 = np.sum(np.multiply(u[: , np.array(rating[: , 0] , dtype='int32')] , v[: , np.array(rating[: , 1] , dtype='int32')]) , axis = 0)
            p2 = p1
            err = rating[: , 2] - p2
            ## for train data , take the regularization into consideration
            regu_part = regu_u * np.sum(np.sum(np.power(u , 2))) + regu_v * np.sum(np.sum(np.power(v , 2)))
            l = sqrt((np.sum(np.power(err , 2)) + regu_part)/rating.shape[0])
            return l
        else:
            p1 = np.sum(np.multiply(u[: , np.array(rating[: , 0] , dtype='int32')] , v[: , np.array(rating[: , 1] , dtype='int32')]) , axis = 0)
            p2 = p1
            err = rating[: , 2] - p2
            l = sqrt(np.sum(np.power(err , 2))/rating.shape[0])
            return l


def print_info(train_rating ,  dim , lr , regu_u , regu_v , epoch , batch_size):
    print '** pmf_batch_training **'
    print 'dim : ',dim,' || lr : ',lr,' || regu_u : ',regu_u,' || regu_v : ',regu_v,' || epoch : ',epoch,' || batch_size :',batch_size
    print 'train user number : ' , np.amax(train_rating[:,0])
    print 'train item number : ' , np.amax(train_rating[:,1])
    print 'train rating number : ' , train_rating.shape[0]



## mini-batch trainning : using SGD
def pmf_batch(train_rating ,  dim , lr , regu_u , regu_v , epoch , batch_size):

    print_info(train_rating ,  dim , lr , regu_u , regu_v , epoch , batch_size)

    starttime = datetime.datetime.now()
    u = np.random.normal(0 , sigma_u , (dim , int(np.amax(train_rating[:,0]))+1))
    v = np.random.normal(0 , sigma_v , (dim , int(np.amax(train_rating[:,1]))+1))

    batch_num = int(train_rating.shape[0] / batch_size)
    train_loss = []
    for i in range(epoch):
        shuffled_order = np.arange(train_rating.shape[0])
        np.random.shuffle(shuffled_order)
        for batch_index in range(batch_num):
            batch_content = np.array(train_rating[shuffled_order[batch_size * batch_index : batch_size * (batch_index + 1)] ,:])
            user_id = np.array(batch_content[:,0], dtype='int32')
            item_id = np.array(batch_content[:,1], dtype='int32')
            user_feature = u[: , user_id]
            item_feature = v[: , item_id]
            p = np.sum(np.multiply(user_feature , item_feature) , axis=0)
            predict = p
            err = batch_content[:,2] - predict
            u_gradient = np.multiply(err , item_feature) - regu_u * user_feature
            v_gradient = np.multiply(err , user_feature) - regu_v * item_feature

            for a in range(batch_size):
                u[: , user_id[a]] = u[: , user_id[a]] + lr * u_gradient[: , a]/batch_size
                v[: , item_id[a]] = v[: , item_id[a]] + lr * v_gradient[: , a]/batch_size

        train_l = compute_loss('train' , train_rating , u , v , regu_u , regu_v)
        train_loss.append(train_l)
        print 'epoch : ' , i
        print 'loss on train set : ',train_l

    endtime = datetime.datetime.now()
    print 'batchtrain -- total execute time : (seconds)'
    print (endtime - starttime).seconds

## plot the loss function
    plt.plot(range(epoch) , train_loss ,color = 'red' , label = 'train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()
    na = 'epoch_'+str(epoch)+'.jpg'
    plt.savefig(na)
    return u , v
