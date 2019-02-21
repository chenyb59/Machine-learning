import sys
sys.path.append('./')
from libsvm.python.svmutil import *
from libsvm.python.svm import *

label = {0:'business' , 1:'computers' , 2:'culture-arts-entertainment' , 3:'education-science' , 4:'engineering' , 5:'health' , 6:'politics-society' , 7:'sports'}

# y, x = svm_read_problem('train.txt')
# yv, xv = svm_read_problem('validation.txt')

y, x = svm_read_problem('train_output.txt')
yt, xt = svm_read_problem('test_output.txt')

## using the libsvm/tools/grid.py to find the parameter
parameter = '-c 32.0 -g 0.015625'
print 'training : '
svm_model = svm_train(y, x ,parameter)

# print 'predict on validation set :'
# v_label, v_acc, v_val = svm_predict(yv , xv , svm_model)
#print p_label

print 'predict on test set :'
t_label, t_acc, t_val = svm_predict(yt , xt , svm_model)
f = open('test_predict.txt','w')
for a in t_label:
    f.write(label.get(int(a)))
    f.write('\n')
f.close()

svm_save_model('model_file', svm_model)
