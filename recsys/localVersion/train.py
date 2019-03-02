import sys
from PMFModel import *
import numpy as np
import matplotlib.pyplot as plt
import math

def main(argv):
    data = []
    trainset = np.loadtxt(argv[1], dtype='int', delimiter='\t')
    testset = np.loadtxt(argv[2], dtype='int', delimiter='\t')

    recsys = PMF(trainset)
    recsys.print_info()

    max_lr = 1
    min_lr = 0.1
    step_lr = 0.1
    max_regu = 0.1
    min_regu = 0.01
    step_regu = 0.01
    min_rmse, opt_epoch, opt_lr, opt_regu = recsys.grid_search(testset, dim, max_lr, min_lr, step_lr, max_regu, min_regu, step_regu)
    # print 'min_rmse : {}'.format(min_rmse)
    # print 'opt_epoch : {}'.format(opt_epoch)
    # print 'opt_lr : {}'.format(opt_lr)
    # print 'opt_regu : {}'.format(opt_regu)

    dim = 4
    regu = 0.01
    epoch = 50
    batch_size = 32
    lr = 0.1
    momentum = 0.1

    # SGD
    recsys.train(dim , lr , regu , epoch)

    # batch train
    recsys.train(dim , lr , regu , epoch , batch_size)

    ## momentum train
    recsys.train(dim , lr , regu , epoch ,1, momentum)

    print recsys.test(testset)
    recsys.save_matrix('user_matrix.npy' , 'item_matrix.npy')
    # user_feature = np.load("user_matrix.npy")



if __name__ == "__main__":
    main(sys.argv)
