import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel("Q1.xlsx")
X = df.drop("Y", axis=1)
Y = df["Y"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=42)

# dataset ready

# (a) closed form solution Q∗ =(X.T@X)^(−1)@X.T@y

def closed_sol(X,Y):
    Q = np.linalg.inv(X.T@X)@(X.T@Y)
    return Q

q_train = closed_sol(x_train,y_train)

def loss_fn(x,y,q):
    y_hat = x@q

    difference = (y - y_hat)**2
    error = 0
    for i in difference:
        error += i
    error /= len(x)

    return error

tr_error = loss_fn(x_train,y_train,q_train)
v_error = loss_fn(x_val,y_val,q_train)
te_error = loss_fn(x_test,y_test,q_train)

print("train_error: ",tr_error)
print("validation_error: ",v_error)
print("test_error: ",te_error)

# (b) Linear regression

def gradient_descent(x,y,r):
    x = np.array(x)
    y = np.array(y)

    w = np.zeros(8)
    b = np.array([0])

    der = np.zeros(8)
    for i in range(w):
        der[i] = np.mean(-2*(x[:,i]@(y - x@w.T - b)))
    
    for i in range(len(w)):
        w[i] -= r*der[i]

    # continue for b, w over



