import numpy as np
from math import sqrt

def generate_matrix(a,b):
  return np.random.normal(loc=0,scale=1,size=(a,b))

def frobenius_norm(a):
  norm = sqrt(np.sum(np.square(a)))
  return norm

frob_norm = frobenius_norm(generate_matrix(6,8))
print("Frobenius norm:",frob_norm)