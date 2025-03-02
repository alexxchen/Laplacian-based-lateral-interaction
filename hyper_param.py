import torch
import torch.nn.functional as F
from functools import partial
from utils.read_data import *

class param_base():
    def __init__(self):
        self.lr = 0.001
        self.round = 5

        self.proto = False
        self.resume = False


class mnist_hyper_param(param_base):
    def __init__(self):
        super(mnist_hyper_param, self).__init__()
        self.data_name = 'MNIST'
        self.channel = 1
        self.h = 28
        self.w = 28

        self.DIM = 2
        self.DIM_NUM = 10
        self.input_dim =784

        self.total_epoch = 100
        self.drop_epoch = [50, 75]
        self.drop_rate = 0.1

        self.decay = 0.92

        self.reg_s = 'sparse'
        self.sparse_reg = 0.004

        self.smooth_reg = 0.1
        self.reg = 'smooth'

        self.trainloader, self.testloader = read_mnist(100)
