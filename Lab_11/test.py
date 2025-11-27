from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

train_data = datasets.MNIST(root="data",train=True,transform=ToTensor(),download=True)
test_data = datasets.MNIST(root="data",train=False,transform=ToTensor(),download=True)

print(train_data.data.shape)
print(test_data.data.shape)

loaders = {
    "train" : DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1),
    "test" : DataLoader(test_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1)
}

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(self.conv2_drop(F.max_pool2d(self.conv2(x),2)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x
    
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epochs):
    model.train()
    for batch_idx,(data,target) in enumerate(loaders['train']):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx%20 == 0:
            print(f"train epoch: {epochs}|[{batch_idx*len(data)}/{len(loaders['train'].dataset)} ({100*batch_idx/len(loaders['train']):.0f}%)]\t{loss.item():.6f}")

def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data,target in loaders["test"]:
            data,target = data,target
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(loaders['test'].dataset)
    print(f"\nTest set: average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")


if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()

model.eval()

data,target  = test_data[0]

data = data.unsqueeze(0)

output = model(data)

prediction = output.argmax(dim=1, keepdim=True).item()

print(f"prediction: {prediction}")

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap='gray')
plt.show()