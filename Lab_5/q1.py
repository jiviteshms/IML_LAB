import numpy as np
import cvxpy as cp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("Q1.csv")
X = df.drop("Y", axis=1).values
Y = df["Y"].values

# Y=np.expand_dims(Y,axis=1)
Y=np.where(Y==-1,0,Y)
print(Y.shape)

# one = np.ones((X.shape[0],1))
# X = np.column_stack((X,one))
print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=72)

svm = SVC(kernel="linear")
svm.fit(x_train,y_train)

par = svm.coef_
print("parameters: ",par)

DecisionBoundaryDisplay.from_estimator(
        svm,
        x_train,
        response_method="predict",
        alpha=0.8,
    )

plt.scatter(x_train[:,0],x_train[:,1])
plt.show()

train_predict = svm.predict(x_train)
# print(train_predict)

test_predict = svm.predict(x_test)
# print(test_predict)

def accuracy(y,y_pred):
    count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            count += 1
    return count/len(y)

train_acc = accuracy(y_train,train_predict)
print("training accuracy: ",train_acc)

test_acc = accuracy(y_test,test_predict)
print("test accuracy: ",test_acc)

# b
w = svm.coef_[0]              
b = svm.intercept_[0]           
xx = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)

# Decision boundary
yy = -(w[0]*xx + b)/w[1]
yy_plus = -(w[0]*xx + b - 1)/w[1]
yy_minus = -(w[0]*xx + b + 1)/w[1]

plt.scatter(x_train[:,0], x_train[:,1], label="Train")
plt.scatter(x_test[:,0], x_test[:,1], label="Validation")
plt.plot(xx, yy, 'k-', label="Boundary")
plt.plot(xx, yy_plus, 'k--', xx, yy_minus, 'k--', label="Margins")
plt.scatter(svm.support_vectors_[:,0], svm.support_vectors_[:,1], s=120,
            facecolors='none', edgecolors='g', label="Support Vectors")
plt.legend(); plt.show()




