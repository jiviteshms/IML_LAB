# Name : Jivitesh M S
# Roll no : b24me1039

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

df = pd.read_csv("dataset.csv")

x1 = np.array(df["x1"])
x2 = np.array(df["x2"])
x3 = np.array(df["x3"])
x4 = np.array(df["x4"])

X = np.vstack([x1,x2,x3,x4])
Y = np.array(df["class"])

# glass = np.array(df["class"])

iris_plant={}
def str_to_num(iris):
    if iris not in iris_plant: iris_plant[iris]=len(iris_plant)+1
    return iris_plant[iris]
df["y_encode"]=df["class"].apply(str_to_num)
Y=df["y_encode"].values.astype(int).ravel()

# a

means = np.mean(X, axis=1)
means = means.reshape(4,1)
print("means: ",means)

mean_centered_x = X - means
print(mean_centered_x)
print(mean_centered_x.shape)

# b

n = len(x1)
covariance_matrix = (1/n)*np.dot(mean_centered_x,mean_centered_x.T)
print("Covariance matrix:\n ",covariance_matrix)

vals, vecs = np.linalg.eigh(covariance_matrix)

vals = vals[::-1]
vals = np.diag(vals)

vecs = vecs[:,::-1]

print("eigen value matrix:\n",vals)
print(f"eigen vectors:\n",vecs)

# c
W = np.vstack((vecs[:,0].T,vecs[:,1].T))
print("projection matrix: \n",W)

y_i = np.dot(W,mean_centered_x)
print("Lower dimension rep: \n",y_i)

# d
U = W.T
x_i_bar = np.dot(U,y_i) 
print("reconstructed point: \n",x_i_bar)

# e
rec_error = np.mean(np.square(mean_centered_x - x_i_bar))
print("reconstruction error: \n", rec_error)

# f
class SpectralClustering:
    def _init_(self): 
        self.labels=None
    def fit(self,X,sigma=1.0,p=3):
        X=X.T
        n=X.shape[0]
        dist=np.linalg.norm(X[:,np.newaxis]-X,axis=2)
        W=np.exp(-dist*2/(sigma*2))
        D=np.diag(W.sum(axis=1))
        L=D-W
        D_inv_sqrt=np.diag(1.0/np.sqrt(W.sum(axis=1)))
        L_norm=D_inv_sqrt@L@D_inv_sqrt
        eigvals,eigvecs=np.linalg.eigh(L_norm)
        idx=np.argsort(eigvals)[:p]
        self.H=eigvecs[:,idx]
        self.H=self.H/np.linalg.norm(self.H,axis=1,keepdims=True)
        self.model=KMeans(n_clusters=p)
        self.model.fit(self.H)
        self.labels=self.model.labels_+1
        return self

def cluster_accuracy(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred)
    row_ind,col_ind=linear_sum_assignment(-cm)
    matched=cm[row_ind,col_ind].sum()
    return matched/y_true.size*100.0

model_before=SpectralClustering()
model_before.fit(X,sigma=1.0,p=3)
labels_before=model_before.labels
acc_before=cluster_accuracy(Y,labels_before)

print(acc_before)

def pca_project(X,V,k):
    Vk=V[:,:k]
    W=Vk.T
    U=Vk
    Y=W@X
    X_hat=U@Y
    return W,U,Y,X_hat

acc_after={}
labels_after={}
for k in [1,2,3]:
    _,_,Y_proj,_=pca_project(X,vecs,k)
    model_after=SpectralClustering()
    model_after.fit(Y_proj,sigma=1.0,p=3)
    labels_after[k]=model_after.labels
    acc_after[k]=cluster_accuracy(Y,model_after.labels)
    print(f"Spectral clustering accuracy after PCA (k={k}): {acc_after[k]:.2f}%")

recon_errors={}
for k in [1,2,3]:
    _,_,_,X_hat=pca_project(X,vecs,k)
    err=np.sum(np.linalg.norm(X-X_hat,axis=0)**2)/n  
    recon_errors[k]=err
    print(f"Reconstruction error (k={k}): {err:.6f}")

# g
_,_,Y_k2,_=pca_project(X,vecs,2)
X_vis=Y_k2.T
plt.scatter(X_vis[:,0],X_vis[:,1],c=Y,cmap='viridis',s=40)
plt.title("Original data (ground truth)")
plt.show()
plt.scatter(X_vis[:,0],X_vis[:,1],c=labels_before,cmap='tab10',s=40)
plt.title(f"Spectral clustering before PCA\nAcc: {acc_before:.2f}%")
plt.show()
plt.scatter(X_vis[:,0],X_vis[:,1],c=labels_after[2],cmap='tab10',s=40)
plt.title(f"Spectral clustering after PCA (k=2)\nAcc: {acc_after[2]:.2f}%")
plt.show()

# Summary 
for k in [1,2,3]: print(f"k={k}: recon_error = {recon_errors[k]:.6f}, acc_after = {acc_after[k]:.2f}%")
print(f"Acc before PCA: {acc_before:.2f}%")
