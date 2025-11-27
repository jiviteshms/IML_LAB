# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
df = pd.read_excel("Q1.xlsx")
X = df.drop("Y", axis=1).values
Y = df["Y"].values

xmin=np.min(X,axis=0)
xmax=np.max(X,axis=0)

X=(X-xmin)/(xmax-xmin)
one = np.ones((X.shape[0],1))

X = np.column_stack((X,one))

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=42)

print(x_train)

# %% [markdown]
# a)

# %%
# (a) closed form solution Q∗ =(X.T@X)^(−1)@X.T@y

def closed_sol(X,Y):
    Q = np.linalg.pinv(X.T@X)@(X.T@Y) 
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

# %%
import matplotlib.pyplot as plt

# %%
def linear_regression(x,y,lr,epoch):
    w = np.zeros(9)
    m = x.shape[0]
    print(m)
    grad_mag = []

    for n in range(epoch):
        y_hat = x @ w
        grad = (-2/m) * (x.T @ (y - y_hat))  # Gradient vector
        grad_mag.append(np.linalg.norm(grad))  # Store magnitude
        w = w - lr * grad

        
        error = np.mean((w@x.T - y)**2)
        print(f"epoch {n+1}, error:  ", error)

    return w,grad_mag

# %%
epochs = 100000
w,grad_mag = linear_regression(x_train,y_train,0.1,epochs)

# %%
print(w)
print(q_train)

# %%
def mse(x, y, w):
    return np.mean((x @ w - y)**2)

gd_train_error = mse(x_train, y_train, w)
gd_val_error = mse(x_val, y_val, w)
gd_test_error = mse(x_test, y_test, w)

print("Gradient Descent train_error: ", gd_train_error)
print("Gradient Descent validation_error: ", gd_val_error)
print("Gradient Descent test_error: ", gd_test_error)

# %%
epochs = np.arange(epochs)
plt.plot(epochs,grad_mag)

# %%



