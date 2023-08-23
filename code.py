import numpy as np

def E(v, h, b, c, W):
    return np.exp(np.dot(b, v)+np.dot(c, h)+np.dot(v, np.matmul(W, h)))
# number of visible and hidden units
hu_size = 3
vu_size = 3

sample_space = np.array([[np.random.random() for x in range(0, vu_size)] for x in range(0, 20)])

hidden_units = np.array([1 for x in range(0, hu_size)])
b, c = np.array([np.random.random() for x in range(0, vu_size)]), np.array([np.random.random() for x in range(0, hu_size)])

#size of the matrix must be 
W = np.array([[np.random.random() for x in range(0, hu_size)] for x in range(0, vu_size)])

def log_likelihood(dataset):
    norm = np.linalg.norm(dataset)
    sum = 0
    for v in dataset:
        sum += np.log(E(v, hidden_units, b, c, W))
    return sum/norm

print(log_likelihood(sample_space))
