import train as tr
import test as te


classifier=tr.SVM_train()
acc=te.SVM_test(classifier)
print(acc)