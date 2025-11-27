import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dataset.txt")

X = np.array(df["X"])
Y = np.array(df["Y"])
C = np.array(df["Class"])

n = len(X)

pairwise_dist = list()

# calculating pairwise distances
for i in range(n):
    for j in range(n):
        pairwise_dist.append(np.sqrt(np.square(X[i]-X[j]) + np.square(Y[i]-Y[j])))

# calculating W

W = np.array(pairwise_dist).reshape((n,n))

temp = np.sort(np.array(pairwise_dist))

sigma = np.median(np.array(pairwise_dist))

W = np.exp(W/sigma)

# finding the diagonal matrix
diagonal_elements = np.sum(W,axis=1)
D = np.diag(diagonal_elements)

# Laplacian matrix
L = D - W

# eigens of L
vals,vecs = np.linalg.eigh(L)

print(f"eigenvalues:\n {vals}\n")
print(f"eigenvectors:\n {vecs}\n")
print(vecs.shape)

h = vecs[:,:2]

print(h,h.shape)
plt.scatter(h[:,0],h[:,1])
print(h[:,0].shape)

plt.show()

k = 2

initial_1 = np.random.randint(len(h[0]))
initial_2 = np.random.randint(len(h[0]))

centres = [(h[0][initial_1],h[1][initial_1]),(h[0][initial_2],h[1][initial_2])]

print("initial random centres:",centres)

def distance(p1,p2):
    dist = np.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
    return dist

def average(glass):
    x_sum, y_sum = 0,0
    for i in range(len(h[0])):
        if predictions[i] == glass:
            x_sum += h[0][i]
            y_sum += h[1][i]

    x_avg = x_sum/len(h[0])
    y_avg = y_sum/len(h[0])

    new_center = (x_avg,y_avg)

    return new_center


# k-means ckustering algo
predictions = np.random.randint(1,3,size=len(h[0]))

iterations = 10
for j in range(iterations):
    for i in range(len(h[0])):
        point = (h[0][i],h[1][i])

        if distance(point,centres[0]) < distance(point,centres[1]):
            predictions[i] = 1
        else:
            predictions[i] = 2

    new_centre_1 = average(1)
    new_centre_2 = average(2)

print("centre 1:",new_centre_1)
print("centre 2:",new_centre_2)

plt.scatter(h[0][predictions==1],h[1][predictions==1],color="green")
plt.scatter(h[0][predictions==2],h[1][predictions==2],color="red")
plt.show()

# performance
correct = 0
for i in range(len(predictions)):
    if predictions[i] == C[i]:
        print(predictions[i],C[i])
        correct += 1
print(correct)
accuracy = correct/len(predictions)
print(f"accuracy: {accuracy*100} %")