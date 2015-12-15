from svmutil import *
# Read data in LIBSVM format
y, x = svm_read_problem('../heart_scale')
y, x = svm_read_problem('../heart_scale')
m = svm_train(y[:200], x[:200], '-c 4 -b 1')
svm_save_model('heart_scale.model',m)
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m,'-b 1')

for p in p_val:
    print (p)
