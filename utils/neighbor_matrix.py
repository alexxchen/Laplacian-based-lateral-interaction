import torch
import numpy as np

ORDER = 1

def neighbor_matrix(num_each_dim, dim, order=1, out_dist=False):
	dist = np.zeros((num_each_dim**dim, num_each_dim**dim))
	for i in range(num_each_dim**dim):
		for j in range(num_each_dim**dim):
			dist[i,j] = np.sum((get_index(i, num_each_dim, dim) - get_index(j, num_each_dim, dim))**2)

	threshold = order**2 
	mask = ((0 < dist) & (dist <= threshold)).astype(float)
	if out_dist == True:
		return mask, dist
	else:
		return mask

def get_index(i, num_each_dim, dim):
	index = []
	remain = i
	for j in range(1, dim):
		x = (remain // (num_each_dim**(dim - j)))
		remain = remain % (num_each_dim**(dim - j))
		index.append(np.expand_dims(x, axis=-1))
	index.append(np.expand_dims(remain, axis=-1))
	index = np.concatenate(index, axis=-1)
	return index

class Laplacian_reg():
    def __init__(self, num_each_dim, dim, device):
        self.W = neighbor_matrix(num_each_dim, dim)
        self.D = np.diag(np.sum(self.W, axis= -1))
        self.Laplacian = torch.from_numpy(self.D - self.W).float()
        self.Laplacian = self.Laplacian.to(device)
        
    def __call__(self, inputs):
        reg = torch.sum(torch.mul(torch.matmul(inputs, self.Laplacian), inputs), dim=1)
        reg = torch.mean(reg)
        return reg

