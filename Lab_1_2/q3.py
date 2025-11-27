import numpy as np
# a

f = open("iris.txt", "r")
n = len(f.readlines())
f.seek(0)

list = []
for i in range(n):
  f.readline()

  line = []

  for i in f.readline().split(","):
    try:
      line.append(float(i))
    except:
      pass
  if len(line):
    list.append(line)

f.close()

list = np.array(list)
print(list)

cov_matrix = np.cov(list.T)
print("\n THe covariance matrix is: \n",cov_matrix)

# b

print("most",np.where(cov_matrix == np.max(cov_matrix)))
print("least",np.where(cov_matrix == np.min(cov_matrix)))

# c

feature_1 = []
feature_2 = []
feature_3 = []
feature_4 = []

for i in list:
  feature_1.append(i[0])
  feature_2.append(i[1])
  feature_3.append(i[2])
  feature_4.append(i[3])

mean_1 = np.mean(feature_1)
mean_2 = np.mean(feature_2)
mean_3 = np.mean(feature_3)
mean_4 = np.mean(feature_4)

print("1:",mean_1,"2:",mean_2,"3:",mean_3,"4:",mean_4)

var_1 = np.var(feature_1)
var_2 = np.var(feature_2)
var_3 = np.var(feature_3)
var_4 = np.var(feature_4)

print("1:",var_1,"2:",var_2,"3:",var_3,"4:",var_4)

# d

val,vec = np.linalg.eigh(cov_matrix)
print("val",val)