#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from math import *
import datetime


sigma_u = 0.01
sigma_v = 0.01

## mini-batch trainning
def pmf_batch(train_rating , test_rating, dim , lr , regu_u , regu_v , epoch , batch_size):

    def compute_loss():
        p1 = np.sum(np.multiply(u[: , np.array(train_rating[: , 0] , dtype='int32')] , v[: , np.array(train_rating[: , 1] , dtype='int32')]) , axis = 0)
        p2 = 1/(1 + np.exp(-1 * p1))
        train_err = train_rating[: , 2] - p2 - mean_rating
        train_l = sqrt(np.sum(np.power(train_err , 2))/train_rating.shape[0])
        train_loss.append(train_l)
        print 'train loss : ',train_l


        if (np.amax(test_rating[: , 0]) > np.amax(train_rating[:,0])) | (np.amax(test_rating[: , 1]) > np.amax(train_rating[:,1])):
            print 'there is userid or itemid exceed the trainset max id.'
        else:
            p1 = np.sum(np.multiply(u[: , np.array(test_rating[: , 0] , dtype='int32')] , v[: , np.array(test_rating[: , 1] , dtype='int32')]) , axis = 0)
            p2 = 1/(1 + np.exp(-1 * p1))
            test_err = test_rating[: , 2] - p2 - mean_rating
            test_l = sqrt(np.sum(np.power(test_err , 2))/test_rating.shape[0])
            test_loss.append(test_l)
            print 'test loss : ',test_l

    print '** pmf_batch_training **'
    print 'dim : ',dim,' || lr : ',lr,' || regu_u : ',regu_u,' || regu_v : ',regu_v,' || epoch : ',epoch,' || batch_size :',batch_size
    print 'max user number : ' , np.amax(train_rating[:,0])
    print 'max item number : ' , np.amax(train_rating[:,1])
    print 'train_rating number : ' , train_rating.shape[0]
    print 'max user number : ' , np.amax(test_rating[:,0])
    print 'max item number : ' , np.amax(test_rating[:,1])
    print 'test_rating number : ' , test_rating.shape[0]


    starttime = datetime.datetime.now()

    u = np.random.normal(0 , sigma_u , (dim , int(np.amax(train_rating[:,0]))+1))
    v = np.random.normal(0 , sigma_v , (dim , int(np.amax(train_rating[:,1]))+1))

    #u = np.ones((dim , int(np.amax(train_rating[:,0]))+1))
    #v = np.ones((dim , int(np.amax(train_rating[:,1]))+1))

    batch_num = int(train_rating.shape[0] / batch_size)
    mean_rating = np.mean(train_rating[: , 2])



    train_loss = []
    test_loss = []
    for i in range(epoch):

        for batch_index in range(batch_num):

            batch_content = np.array(train_rating[batch_size * batch_index : batch_size * (batch_index + 1) ,:])

            user_id = np.array(batch_content[:,0], dtype='int32')
            item_id = np.array(batch_content[:,1], dtype='int32')
            user_feature = u[: , user_id]
            item_feature = v[: , item_id]



            p = np.sum(np.multiply(user_feature , item_feature) , axis=0)


            predict = 1/(1 + np.exp(-1 * p))


            err = batch_content[:,2] - predict - mean_rating


            u_gradient = np.multiply(err , item_feature) - regu_u * user_feature
            v_gradient = np.multiply(err , user_feature) - regu_v * item_feature


            for a in range(batch_size):
                u[: , user_id[a]] = u[: , user_id[a]] + lr * u_gradient[: , a]/batch_size
                v[: , item_id[a]] = v[: , item_id[a]] + lr * v_gradient[: , a]/batch_size




        print 'epoch : ' , i
        compute_loss()


    endtime = datetime.datetime.now()
    print 'batchtrain -- total execute time : (seconds)'
    print (endtime - starttime).seconds

## plot the loss function
    plt.plot(range(epoch) , train_loss ,color = 'red' , label = 'train loss')
    plt.plot(range(epoch) , test_loss , color = 'blue' , label = 'test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()
    plt.savefig("pmf-lr=0.1-re-0.05.jpg")
    return u , v
