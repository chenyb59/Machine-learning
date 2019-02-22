from pmf_code import *
#from pmf_batchtrain import *
from pmf_batchtrain_momentum import *
import numpy as np
import matplotlib.pyplot as plt
import math

dim = 10
learning_rate = 0.1
lamda_u = 0.05
lamda_v = 0.05
epoch_num = 350
batch_size = 32

## normalize to [0,1] to improve stability
def normalize_rating(arr):
    max = np.amax(arr[:,2])
    min = np.amin(arr[:,2])
    arr[:,2] = (arr[:,2] - min) / (max - min)
    return arr



def main():
    train_path = 'train_100.txt'
    test_path = 'test_100.txt'

    train_data = normalize_rating(np.loadtxt(train_path))
    test_data = normalize_rating(np.loadtxt(test_path))


    #user_feature , music_feature = pmf(train_data ,test_data , dim , learning_rate ,lamda_u , lamda_v , epoch_num)
    user_feature , music_feature = pmf_batch(train_data ,test_data , dim , learning_rate ,lamda_u , lamda_v , epoch_num , batch_size)






if __name__ == "__main__":
    main()
