import numpy as np
from .neighbor_matrix import neighbor_matrix

def neighbor_d(n, dim):
    # this is different from neighbor matrix in Laplacian matrix
	dist = np.zeros((n**dim, n**dim))
	for i in range(n**dim):
		for j in range(n**dim):
			dist[i,j] = np.sum((get_index(i, n, dim) - get_index(j, n, dim))**2)

	return dist

def get_index(i, n, dim):
    # for n number each dimension, get the location of i
	index = []
	remain = i
	for j in range(1, dim):
		x = (remain // (n**(dim - j)))
		remain = remain % (n**(dim - j))
		index.append(np.expand_dims(x, axis=-1))
	index.append(np.expand_dims(remain, axis=-1))
	index = np.concatenate(index, axis=-1)
	return index

def gaussian(dist, sigma):
	guassian = np.exp(- dist / (sigma**2))
	return guassian

def make_one_hot(data, class_num):
	return (np.eye(class_num)[data]).astype(np.float64)

class SOM_loss():
    def __init__(self, num_each_dim, dim):
        dist = neighbor_d(num_each_dim, dim)
        self.gaussian_kernel = gaussian(dist, 1)
        self.n_clusters = num_each_dim**dim

    def __call__(self, dictionary, inputs):
        # (batch_size, dict_size)
        SE = np.sum((np.expand_dims(inputs, 1) - np.expand_dims(dictionary, 0))**2, axis=-1)
        # (batch_size, dict_size) each row is the loss for each winner (when i is the winner, the loss should at the i place)
        loss = np.matmul(SE, self.gaussian_kernel.T)
        loss = np.min(loss, axis=-1)
        loss = np.mean(loss)
        return loss


    def VQ_loss(self, dictionary, inputs):
        # use the most activated prototype to caculate reconstrution VQ loss
        out = np.sum( (np.expand_dims(inputs, 1) - np.expand_dims(dictionary, 0))**2, axis=-1)
        loss = np.min(out, axis=-1)
        loss = np.mean(loss)
        
        return loss