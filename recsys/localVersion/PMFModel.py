# coding: utf-8
# Author = Chenyb
import numpy as np
import matplotlib.pyplot as plt
from math import *



class PMF(object):
    ## trainset input is a numpy array
    ## rating will be pre-processed using z-score normalization.
    def __init__(self , trainset):
        self.sigma_u = 0.01
        self.sigma_v = 0.01
        self.userid_hash = {}
        self.itemid_hash = {}
        self.r_mean = np.mean(trainset , axis = 0)[2]
        self.r_std = sqrt(np.var(trainset , axis = 0)[2])
        for record in trainset:
            if self.userid_hash.get(record[0]) == None:
                integer_id = len(self.userid_hash)
                self.userid_hash[record[0]] = integer_id
            record[0] = self.userid_hash.get(record[0])
            if self.itemid_hash.get(record[1]) == None:
                integer_id = len(self.itemid_hash)
                self.itemid_hash[record[1]] = integer_id
            record[1] = self.itemid_hash.get(record[1])
            record[2] = (record[2] - self.r_mean)/self.r_std
        self.rating = np.copy(trainset)


    def compute_rmse(self):

        predict_rating = np.sum(np.multiply(self.user_matrix[: , self.rating[:,0]] , self.item_matrix[: , self.rating[:,1]]) , axis = 0)
        err = self.rating[:,2] - predict_rating
        return sqrt(np.sum(np.power(err , 2)) / self.rating.shape[0])

    ## load matrix to files
    def save_matrix(self , f1 , f2):
        np.save(f1, self.user_matrix)
        np.save(f2, self.item_matrix)



    def print_info(self):
        print 'Total user number : {}'.format(len(self.userid_hash))
        print 'Total item number : {}'.format(len(self.itemid_hash))
        print 'Total rating number : {}'.format(self.rating.shape[0])



    def update(self, lr, regu, batch_size = 1, momentum = 0):
        batch_num = int(self.rating.shape[0] / batch_size)
        shuffled_order = np.arange(self.rating.shape[0])
        np.random.shuffle(shuffled_order)
        last_u_grad = 0
        last_v_grad = 0
        for batch_index in range(batch_num):
            shuffled_order[batch_size * batch_index : batch_size * (batch_index + 1)]
            batch_content = np.array(self.rating[shuffled_order[batch_size * batch_index : batch_size * (batch_index + 1)] ,:])
            user_id = np.array(batch_content[:,0], dtype='int32')
            item_id = np.array(batch_content[:,1], dtype='int32')
            user_feature = self.user_matrix[: , user_id]
            item_feature = self.item_matrix[: , item_id]
            predict = np.sum(np.multiply(user_feature , item_feature) , axis=0)
            err = self.rating[shuffled_order[batch_size * batch_index : batch_size * (batch_index + 1)] ,2] - predict
            u_gradient = np.multiply(err , item_feature) - regu * user_feature
            v_gradient = np.multiply(err , user_feature) - regu * item_feature

            for a in range(batch_size):
                self.user_matrix[: , user_id[a]] += lr * u_gradient[: , a]/batch_size + momentum * last_u_grad
                self.item_matrix[: , item_id[a]] += lr * v_gradient[: , a]/batch_size + momentum * last_v_grad
                last_u_grad = u_gradient[: , a]
                last_v_grad = v_gradient[: , a]



    ## Default optimization : SGD ;
    ## momentum and batch training can also be triggered by setting the two parameters : batch_size , momentum.
    def train(self , dim , lr , regu , epoch , batch_size = 1, momentum = 0):
        self.user_matrix = np.random.normal(0 , self.sigma_u , (dim , len(self.userid_hash))).astype('f')
        self.item_matrix = np.random.normal(0 , self.sigma_v , (dim , len(self.itemid_hash))).astype('f')
        rmse = []
        for i in range(epoch):
            self.update(lr, regu, batch_size, momentum)
            r = self.compute_rmse()
            rmse.append(r)
            print 'epoch : {}'.format(i)
            print 'RMSE on train set : {}'.format(r)


    def predict(self , u , v):
        if self.userid_hash.get(u)!=None and self.itemid_hash.get(v)!=None:
            user_feature = self.user_matrix[: , self.userid_hash.get(u)]
            item_feature = self.item_matrix[: , self.itemid_hash.get(v)]
            # print user_feature.shape,item_feature.shape
            return np.sum(np.multiply(user_feature , item_feature) , axis=0)
        else:
            # print 'id not exist in trainset'
            return None


    ## grid_search is used to find the optimal value of lr and regu v for a specific training dataset.
    ## A testset should be designated to pick optimal parameter combination.
    def grid_search(self, testset, dim, max_lr, min_lr, step_lr, max_regu, min_regu, step_regu):
        min_rmse = None
        print '************************'
        print 'Grid search begining ...'
        print '************************'
        for lr in np.arange(min_lr, max_lr + step_lr, step_lr):
            for regu in np.arange(min_regu, max_regu + step_regu, step_regu):
                print 'lr : {0} , regu : {1} '.format(lr,regu)
                self.user_matrix = np.random.normal(0 , self.sigma_u , (dim , len(self.userid_hash))).astype('f')
                self.item_matrix = np.random.normal(0 , self.sigma_v , (dim , len(self.itemid_hash))).astype('f')
                self.update(lr, regu)
                test_rmse = self.test(testset)
                count = 1
                while count==1 or test_rmse < last_test_rmse:
                    print 'epoch {0} : {1}'.format(count, test_rmse)
                    last_test_rmse = test_rmse
                    self.update(lr, regu)
                    test_rmse = self.test(testset)
                    count += 1
                if min_rmse == None or last_test_rmse < min_rmse:
                    min_rmse = last_test_rmse
                    opt_epoch = count
                    opt_lr = lr
                    opt_regu = regu
                print '\n'

        return min_rmse, opt_epoch, opt_lr, opt_regu


    ## Test function is used to evaluate the model on test dataset.
    def test(self , testset):
        err = 0
        count = 0
        for record in testset:
            norm_rating = (record[2] - self.r_mean)/self.r_std
            pred = self.predict(record[0],record[1])
            if pred!=None:
                err +=  pow(norm_rating-pred , 2)
                count += 1
        return sqrt(err/count)
