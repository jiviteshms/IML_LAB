import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

X = torch.tensor(df.drop(columns="label").values, dtype=torch.float32)
Y = torch.tensor(df["label"].values, dtype=torch.float32)

# changing to 0,1
Y[Y == 1] = 0
Y[Y == 2] = 1

# split
x_train, x_testing, y_train, y_testing = train_test_split(X, Y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_testing, y_testing, test_size=0.5, random_state=42)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc

# creating model (5,10,15,10,5)
class Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.model_architecture = nn.Sequential(
            nn.Linear(input_features,5),
            nn.ReLU(),
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,15),
            nn.ReLU(),
            nn.Linear(15,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model_architecture(x)
    
neural_network = Model(input_features=2) # input features is the number of features in x
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(params=neural_network.parameters(), lr=0.01)

epochs = 5000

for i in range(epochs):
    # forward pass
    y_pred = neural_network(x_train).squeeze()

    # compute loss on sigmoid outputs (not rounded)
    loss = loss_fn(y_pred, y_train)

    # accuracy on rounded predictions
    y_pred_class = torch.round(y_pred)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred_class)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # testing
    neural_network.eval()
    with torch.inference_mode():
        test_pred = neural_network(x_test).squeeze()
        test_loss = loss_fn(test_pred, y_test)
        test_pred_class = torch.round(test_pred)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred_class)

    if i % 100 == 0:
        print(f"Epoch: {i} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    

# using the trained model

neural_network.eval()
with torch.inference_mode():
    y_logits_train = neural_network(x_train)
    y_logits_val = neural_network(x_val)
    y_logits_test = neural_network(x_test)

y_pred_train = torch.round(y_logits_train)
y_pred_val = torch.round(y_logits_val)
y_pred_test = torch.round(y_logits_test)

train_cm = confusion_matrix(y_train.numpy(),y_pred_train.numpy())
val_cm = confusion_matrix(y_val.numpy(),y_pred_val.numpy())
test_cm = confusion_matrix(y_test.numpy(),y_pred_test.numpy())

print(train_cm)

def metrics(cm):
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

train_pre, train_rec, train_f1 = metrics(train_cm)
val_pre, val_rec, val_f1 = metrics(val_cm)
test_pre, test_rec, test_f1 = metrics(test_cm)

print(f"training:\nprecesion:{train_pre}\nrecall:{train_rec}\nf1:{train_f1}\n")
print(f"validation:\nprecesion:{val_pre}\nrecall:{val_rec}\nf1:{val_f1}\n")
print(f"testing:\nprecesion:{test_pre}\nrecall:{test_rec}\nf1:{test_f1}\n")

def pre_vs_rec():
    thresholds = torch.arange(1, -0.01, -0.05)

    train_precisions = list()
    train_recalls = list()

    val_precisions = list()
    val_recalls = list()

    test_precisions = list()
    test_recalls = list()

    neural_network.eval()
    with torch.inference_mode():
        base_train = neural_network(x_train)
        base_val = neural_network(x_val)
        base_test = neural_network(x_test)

    for i in thresholds:
        y_pred_train = torch.where(base_train>=i,1,0)
        y_pred_val = torch.where(base_val>=i,1,0)
        y_pred_test = torch.where(base_test>=i,1,0)

        train_cm = confusion_matrix(y_train.numpy(),y_pred_train.numpy())
        val_cm = confusion_matrix(y_val.numpy(),y_pred_val.numpy())
        test_cm = confusion_matrix(y_test.numpy(),y_pred_test.numpy())

        train_pre, train_rec,_ = metrics(train_cm)
        val_pre, val_rec,_ = metrics(val_cm)
        test_pre, test_rec,_ = metrics(test_cm)

        train_precisions.append(train_pre)
        train_recalls.append(train_rec)

        val_precisions.append(val_pre)
        val_recalls.append(val_rec)

        test_precisions.append(test_pre)
        test_recalls.append(test_rec)

    plt.plot(train_recalls[1:],train_precisions[1:])
    plt.title("training precesion vs recall")
    plt.show()

    plt.plot(val_recalls[1:],val_precisions[1:])
    plt.title("validation precesion vs recall")
    plt.show()

    plt.plot(test_recalls[1:],test_precisions[1:])
    plt.title("testing precesion vs recall")
    plt.show()

pre_vs_rec()

# plt.scatter(X[:,0],X[:,1])
# plt.show()

# Decision boundary

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = torch.meshgrid(
    torch.arange(x_min, x_max, 0.1),
    torch.arange(y_min, y_max, 0.1),
    indexing='ij'  # ensures same orientation as numpy’s default
)

with torch.no_grad():
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    Z = neural_network(grid).numpy().reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", vmin=0, vmax=1)
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel("Feature X1")
plt.ylabel("Feature X2")
plt.title("Neural Network Decision Boundary (PyTorch)")
plt.grid(alpha=0.3)
plt.show()