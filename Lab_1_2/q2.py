import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# a

def generate_matrix(a,b):
  return np.random.normal(loc=0,scale=1,size=(a,b))

a = generate_matrix(100,100)
b = a + a.T

val, vec = np.linalg.eigh(b)
final_val = val[::-1]
final_val = np.diag(val)
print(final_val,"\n")

final_vec = vec[:,::-1]
print(final_vec)

# b

def frobenius_norm(a):
  norm = sqrt(np.sum(np.square(a)))
  return norm

def beekay(b,k):
  BK = np.zeros(b.shape)
  val, vec = np.linalg.eigh(b)
  val = val[::-1]
  vec = vec[:, ::-1]
  for i in range(k):
    vi=np.expand_dims(vec[:,i],axis=1) 
    BK = BK + val[i]*vi@vi.T

  # print(BK, "\n")
  return BK

BK = beekay(b,2)  #taking k = 2
frob = frobenius_norm(b - BK)
print("frobenius norm:",frob)

# c

n = 100

k = []
for i in range(1,n+1):
  k.append(i)

frobs = []
for i in range(1,n+1):
  BK = beekay(b,i)
  frobs.append(float(frobenius_norm(b - BK)))

plt.plot(k, frobs)
plt.show()

# d

I = np.eye(100)
print(frobenius_norm(vec@vec.T - I))

# e

sigma_square = np.arange(101)*10**-2

k = [5,10,20,25,30]

for i in k:
  frobs = []
  for j in range(101):
    a = np.random.normal(loc=0,scale=sqrt(sigma_square[j]),size=(100,100))
    b = a + a.T
    BK = beekay(b,i)
    frobs.append(float(frobenius_norm(b - BK)))
    
  plt.plot(sigma_square, frobs)
  
plt.show()

