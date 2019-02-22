#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math

def pmf(rating , dim , lr , regu_u , regu_v , train_num):
    #print np.amax(rating[:,0])
    u = np.random.normal(0 , 0.1 , (dim , int(np.amax(rating[:,0]))+1))
    v = np.random.normal(0 , 0.1 , (dim , int(np.amax(rating[:,1]))+1))


    loss = np.zeros(train_num)
    for i in range(train_num):
        #predict = u.T * v
        for item in rating:
            row = int(item[0])
            column = int(item[1])
            rate = item[2]
            #print row,column,rate
            predict = 1/(1 + math.exp(-1 * np.dot(u[:,row].T , v[:,column])))
            u_gradient = (rate - predict) * v[:,column] - regu_u * u[:,row]
            v_gradient = (rate - predict) * u[:,row] - regu_v * v[:,column]
            loss[i] = loss[i] + ((rate - predict)**2 + np.power(u[:,row] , 2).sum()*regu_u + np.power(v[:,column] , 2).sum()*regu_v)/2

            u[:,row] = u[:,row] + lr * u_gradient
            v[:,column] = v[:,column] + lr * v_gradient
        print i , ' : ' , loss[i]

    plt.plot(range(train_num) , loss)
    plt.title('loss of train data')
    plt.xlabel('train time')
    plt.ylabel('loss')
    plt.show()
    return u , v
