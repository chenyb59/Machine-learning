import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

## code the labels into integer 0-7
label = ['business' , 'computers' , 'culture-arts-entertainment' , 'education-science' , 'engineering' , 'health' , 'politics-society' , 'sports']
labels = np.array(label)
label_encoder = LabelEncoder()
label_int_code = label_encoder.fit_transform(labels)



## preprocess the train and test file :
## 1. collect the word set
## 2. encode every word using one-hot coder
## 3. map every search snippets into the word space

train_in_path= 'train.dat'
train_out_path = 'train_output.txt'
test_in_path= 'test.dat'
test_out_path = 'test_output.txt'


word_set = []
f = open(train_in_path)
line = f.readline()
while line:
    line = line.strip()
    info = line.split(' ')
    for i in range(len(info)-1):
        if info[i] not in word_set:
            word_set.append(info[i])
    line = f.readline()
f.close()

f = open(test_in_path)
line = f.readline()
while line:
    line = line.strip()
    info = line.split(' ')
    for i in range(len(info)):
        if info[i] not in word_set:
            word_set.append(info[i])
    line = f.readline()
f.close()



words = np.array(word_set)
word_encoder = LabelEncoder()
word_int_code = word_encoder.fit_transform(words)
word_onehot_encoder = OneHotEncoder(sparse = False)
word_int_code = word_int_code.reshape(len(word_int_code) , 1)
word_onehot_code = word_onehot_encoder.fit_transform(word_int_code)

for k in range(len(word_set)):
    print k,' : ',word_set[k]


dim = word_onehot_code.shape[1]

f_w = open(train_out_path,'w')
f = open(train_in_path)
line = f.readline()
count = 0
while line:
    count = count + 1
    feature_val = np.zeros((1,dim))
    line = line.strip()
    info = line.split(' ')
    for i in range(len(info)-1):
        feature_val = feature_val + word_onehot_code[word_set.index(info[i]) , :]
    label_val = label_int_code[label.index(info[-1])]
    f_w.write(str(label_val))
    f_w.write(' ')

    for j in range(dim):
        if feature_val[0,j] != 0:
            s_str = str(j+1) + ':' + str(feature_val[0,j])
            f_w.write(s_str)
            f_w.write(' ')
    f_w.write('\n')
    line = f.readline()
f.close()
f_w.close()


f_w = open(test_out_path,'w')
f = open(test_in_path)
line = f.readline()
count = 0
while line:
    count = count + 1
    feature_val = np.zeros((1,dim))
    line = line.strip()
    info = line.split(' ')
    for i in range(len(info)-1):
        feature_val = feature_val + word_onehot_code[word_set.index(info[i]) , :]
    f_w.write('0')
    f_w.write(' ')

    for j in range(dim):
        if feature_val[0,j] != 0:
            s_str = str(j+1) + ':' + str(feature_val[0,j])
            f_w.write(s_str)
            f_w.write(' ')
    f_w.write('\n')
    line = f.readline()
f.close()
f_w.close()
