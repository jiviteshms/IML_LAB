# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Q2.csv")
X = df.drop("Y", axis=1).values
Y = df["Y"].values

Y=np.expand_dims(Y,axis=1)
Y=np.where(Y==-1,0,Y)
print(Y.shape)

one = np.ones((X.shape[0],1))
X = np.column_stack((X,one))
print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=72)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=92)

# %%
def pred(w,x,threshold=0.5):
    pred = 1/(1 + np.exp(-x@w))

    for i in range(len(pred)):
        if pred[i] >= threshold:
            pred[i] = 1
        else:
            pred[i] = 0

    return pred

# %%
w = np.zeros((3,1))
m = x_train.shape[0]

lr = 0.1
epochs = 7300

train_acc = []
val_acc = []
errors = []

for i in range(epochs):
    h = 1/(1 + np.exp(-(x_train@w))) # h is the prediction
    w = w - lr*(1/m)*x_train.T@(h-y_train)

    train_acc.append((pred(w,x_train)==y_train).sum()/len(y_train))
    val_acc.append((pred(w,x_val)==y_val).sum()/len(y_val))

    esp=1e-15
    error = (-1/m)*np.sum(y_train*np.log(h+esp) + (1-y_train)*np.log(1-h+esp))
    errors.append(error)
    print(f"epoch {i+1}, error:  ", error)

print(w)
print(train_acc)
print(val_acc)
   

# %%
epochs = np.arange(epochs)

plt.plot(epochs, errors, label="Training Error", color="red")
plt.show()
plt.plot(epochs, train_acc, label="Train Accuracy", color="blue")
plt.plot(epochs, val_acc, label="Validation Accuracy", color="green")
plt.show()

# %%
train = pred(w,x_train)
val = pred(w,x_val)
test = pred(w,x_test)
print(train)

# %%
def confusion(predictions,y):
    tp = np.sum((predictions==1)&(y==1))
    fp = np.sum((predictions==1)&(y==0))
    tn = np.sum((predictions==0)&(y==0))
    fn = np.sum((predictions==0)&(y==1))

    matrix = np.array([[tp,fp],
                       [fn,tn]])
    
    eps = 1e-15
    precision = tp/(tp+fp+eps)
    recall = tp/(tp+fn+eps)
    f1 = 2*precision*recall/(precision+recall+eps)

    return matrix,precision,recall,f1

# %%
matrix,precision,recall,f1 = confusion(train,y_train)
print(f"train: \n confusion:{matrix}\n precision:{precision}\n recall:{recall}\n f1:{f1}")

# %%
matrix,precision,recall,f1 = confusion(val,y_val)
print(f"val: \n confusion:{matrix}\n precision:{precision}\n recall:{recall}\n f1:{f1}")

# %%
matrix,precision,recall,f1 = confusion(test,y_test)
print(f"test: \n confusion:{matrix}\n precision:{precision}\n recall:{recall}\n f1:{f1}")

# %%
thresholds = np.arange(0,1,0.05)
print(thresholds)

# %%
precisions = []
recalls = []
for i in thresholds:
    train = pred(w,x_train,i)
    matrix,precision,recall,f1 = confusion(train,y_train)
    precisions.append(precision)
    recalls.append(recall)
print(precisions)
print(recalls)

# %%
plt.plot(precisions,recalls,color="blue")
plt.show()

# %%
precisions = []
recalls = []
for i in thresholds:
    val = pred(w,x_val,i)
    matrix,precision,recall,f1 = confusion(val,y_val)
    precisions.append(precision)
    recalls.append(recall)

plt.plot(precisions,recalls,color="blue")
plt.show()

# %%
precisions = []
recalls = []
for i in thresholds:
    test = pred(w,x_test,i)
    matrix,precision,recall,f1 = confusion(test,y_test)
    precisions.append(precision)
    recalls.append(recall)
print(precisions)
print(recalls)

plt.plot(precisions,recalls,color="blue")
plt.show()

# %%
plt.scatter(x_train[:,0],x_train[:,1],c="green")
plt.scatter(x_val[:,0],x_val[:,1],c="blue")
plt.scatter(x_test[:,0],x_test[:,1],c="red")
# plt.scatter(df[df["Y"]==-1]["X1"],df[df["Y"]==-1]["X2"],c="blue")

plt.plot(df["X1"],-(w[2]+w[0]*df["X1"])/w[1])

# %%



