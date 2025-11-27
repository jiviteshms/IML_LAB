import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dataset.txt")

X = np.array(df["X"])
Y = np.array(df["Y"])
C = np.array(df["Class"])

n = len(X)

pairwise_dist = list()

# calculating pairwise dist
for i in range(n):
    for j in range(n):
        pairwise_dist.append(np.sqrt((X[i]-X[j])**2 + (Y[i]-Y[j])**2))

W = np.array(pairwise_dist).reshape((n, n))

sigma = np.median(W[np.triu_indices(n, k=1)])

W = np.exp(-(W**2) / (2 * sigma**2))

np.fill_diagonal(W, 0.0)

diagonal_elements = np.sum(W, axis=1)
D = np.diag(diagonal_elements)
L = D - W

# eigens of L
vals, vecs = np.linalg.eigh(L)
print(f"eigenvalues:\n {vals}")
print(f"eigenvectors shape: {vecs.shape}")

h = vecs[:, :2]    
print("h shape:", h.shape)

k = 2

initial_1 = np.random.randint(len(h))
initial_2 = np.random.randint(len(h))
centres = [(h[initial_1, 0], h[initial_1, 1]), (h[initial_2, 0], h[initial_2, 1])]

print("initial random centres:", centres)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def average(cluster_label, current_centres):
    index = np.where(predictions == cluster_label)[0]
    if len(index) == 0:
        return current_centres[cluster_label - 1]
    x_avg = np.mean(h[index, 0])
    y_avg = np.mean(h[index, 1])
    return (x_avg, y_avg)

predictions = np.random.randint(1, 3, size=len(h))

iterations = 10
for j in range(iterations):
    for i in range(len(h)):
        point = (h[i, 0], h[i, 1])
        if distance(point, centres[0]) < distance(point, centres[1]):
            predictions[i] = 1
        else:
            predictions[i] = 2
    centres[0] = average(1, centres)
    centres[1] = average(2, centres)

print("final centre 1:", centres[0])
print("final centre 2:", centres[1])

# plot clusters in spectral space
plt.scatter(h[predictions == 1, 0], h[predictions == 1, 1], color="green", label="pred 1")
plt.scatter(h[predictions == 2, 0], h[predictions == 2, 1], color="red", label="pred 2")
plt.show()

correct_direct = np.sum(predictions == C)
pred_swap = 3 - predictions 
correct_swapped = np.sum(pred_swap == C)

if correct_swapped > correct_direct:
    predictions = pred_swap
    correct = int(correct_swapped)
else:
    correct = int(correct_direct)

accuracy = correct / len(predictions)
print(f"accuracy: {accuracy*100} %")

# Plot clusters 
plt.scatter(X[predictions == 1], Y[predictions == 1], color="green", label="Cluster 1")
plt.scatter(X[predictions == 2], Y[predictions == 2], color="red", label="Cluster 2")

plt.show()