import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

df = pd.read_csv("dataset.txt")

X = np.array(df["X"])
Y = np.array(df["Y"])
C = np.array(df["Class"])

k = 2

initial_1 = np.random.randint(len(X))
initial_2 = np.random.randint(len(X))

centres = [(X[initial_1],Y[initial_1]),(X[initial_2],Y[initial_2])]

print("initial random centres:",centres)

def distance(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)
    return dist

def average(glass):
    x_sum, y_sum = 0,0
    for i in range(len(X)):
        if predictions[i] == glass:
            x_sum += X[i]
            y_sum += Y[i]

    x_avg = x_sum/len(X)
    y_avg = y_sum/len(X)

    new_center = (x_avg,y_avg)

    return new_center


# k-means ckustering algo
predictions = np.random.randint(1,3,size=len(X))

iterations = 10
for j in range(iterations):
    for i in range(len(X)):
        point = (X[i],Y[i])

        if distance(point,centres[0]) < distance(point,centres[1]):
            predictions[i] = 1
        else:
            predictions[i] = 2

    new_centre_1 = average(1)
    new_centre_2 = average(2)

print("centre 1:",new_centre_1)
print("centre 2:",new_centre_2)

plt.scatter(X[predictions==1],Y[predictions==1],color="green")
plt.scatter(X[predictions==2],Y[predictions==2],color="red")
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
