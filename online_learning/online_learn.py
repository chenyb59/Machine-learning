import sys
import numpy as np
import matplotlib.pyplot as plt
import math


train_path = 'train.dat'
test_path = 'test.dat'

## get the feature number first

f = open(train_path)
line = f.readline()
max_fea = 1
while line:

    line = line.strip()
    info = line.split(' ')
    last_one = info[-1]
    last = last_one.split(':')
    fea = int(last[0])
    if max_fea < fea:
        max_fea = fea
    line = f.readline()
f.close()

print max_fea

## initialize the w vector with all zero
w = [0] * max_fea


## read the train data as a streaming input , each line is processed once

f = open(train_path)
line = f.readline()
while line:

    line = line.strip()
    info = line.split(' ')
    label = int(info[0])
    feature = info[1:]
    s = 0
    dim_col = []
    val_col = []
    for item in feature:
        items = item.split(':')
        dim = int(items[0]) - 1
        val = int(items[1])
        dim_col.append(dim)
        val_col.append(val)
        s = s + w[dim] * val

    if s * label <= 0:

        for j in range(len(dim_col)):
            w[dim_col[j]] = w[dim_col[j]] + val_col[j] * label

    line = f.readline()
f.close()



## test the perceptron on test data : stote the result of w * x in a list

f = open(test_path)
line = f.readline()
positive = 0
negative = 0
labellist = []
result = []

while line:
    line = line.strip()
    info = line.split(' ')
    label = int(info[0])
    if label == 1:
        positive = positive + 1
    else:
        negative = negative + 1
    labellist.append(label)
    feature = info[1:]
    s = 0
    dim_col = []
    val_col = []
    for item in feature:
        items = item.split(':')
        dim = int(items[0]) - 1
        val = int(items[1])
        s = s + w[dim] * val
    result.append(s)
    line = f.readline()
f.close()



## define a threshold , if w * x > threshold , predicted label = 1 ; else , predicted label = -1
## gradually alter the threshold , see the change of fp-rate and tp-rate

threshold = list(set(result))
threshold.sort()

fp = []
tp = []
false_n_list = []
false_p_list = []

for m in range(len(threshold)):
    false_n = 0
    false_p = 0
    for i in range(len(labellist)):
        if result[i] > threshold[m]:
            predict = 1
        else:
            predict = -1

        if predict * labellist[i] < 0:
            if labellist[i] == 1:
                false_n = false_n + 1
            else:
                false_p = false_p + 1

    false_n_list.append(false_n)
    false_p_list.append(false_p)
    fp_rate = float(false_p) / negative
    tp_rate = float(positive - false_n) / positive
    fp.append(fp_rate)
    tp.append(tp_rate)

plt.plot(fp , tp )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## the optimal (fp,tp) pair is (0,1) , so find the point on ROC curve which is most close to (0,1)

min_dist = 100
for k in range(len(fp)):
    dist = pow(fp[k],2) + pow((tp[k]-1) , 2)
    if dist < min_dist:
        min_dist = dist
        index = k
best_threshold = threshold[index]
print index , best_threshold

## calculate the precision , recall , F1 score under the best threshold

precision = float(positive - false_n_list[index]) / (positive - false_n_list[index] + false_p_list[index])
recall = tp[index]
accuracy = float(positive - false_n_list[index] + negative - false_p_list[index]) / (positive + negative)
f1_score = 2 / (1 / precision + 1 / recall)
print 'the final best result : '
print 'precision : ',precision
print 'recall : ', recall
print 'F1 score : ', f1_score
