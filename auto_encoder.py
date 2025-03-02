import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import imsave


from utils.neighbor_matrix import Laplacian_reg
from utils.som_loss import SOM_loss

def sparseness(x, order):
    lenth = torch.sum(torch.abs(x)**order, dim=-1)**(1/order)
    return torch.mean(lenth)

def make_one_hot(data, class_num):
	return (np.eye(class_num)[data]).astype(np.float64)

class Net(torch.nn.Module):
    def __init__(self, in_h, in_w, channel, hid_num):
        super(Net, self).__init__()

        self.decoder = nn.Linear(hid_num, in_h*in_w*channel, bias=False)

        self.hid_num = hid_num
        self.h = in_h
        self.w = in_w
        self.c = channel

    def forward(self, x):

        x = x.view(x.size(0), -1)

        code = torch.sum( (torch.unsqueeze(x, 1) - torch.unsqueeze(self.decoder.weight.T, 0))**2, axis=-1)
        index = np.argmin(code.cpu().detach().numpy(), axis=-1)
        code = torch.Tensor(make_one_hot(index, self.hid_num)).to('cuda')
        code = code.detach()

        reconst = self.decoder(code)
        reconst = reconst.view(-1, self.c, self.h, self.w)
        return reconst, code


class AE():
    def __init__(self, param, logger):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Net(param.h, param.w, param.channel, param.DIM_NUM**param.DIM).to(self.device)

        self.train_loader = param.trainloader
        self.test_loader = param.testloader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=param.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=param.drop_epoch, gamma=param.drop_rate)
        self.criterion = torch.nn.MSELoss()
        self.reg = Laplacian_reg(param.DIM_NUM, param.DIM, self.device)

        self.som_loss = SOM_loss(param.DIM_NUM, param.DIM)

        self.param = param
        self.logger = logger

        self.smooth_reg = self.param.smooth_reg
        self.sparse_reg = self.param.sparse_reg

    def train(self):
        self.model.train()
        MSE_loss = 0
        VQ_loss = 0
        Sparse_loss = 0
        Laplacian_loss = 0
        SL = 0
        Code_len = 0

        self.smooth_reg = self.smooth_reg * self.param.decay
        
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            
            self.optimizer.zero_grad()
            reconst, code = self.model(inputs)
            MSE = self.criterion(reconst, inputs)

            VQ = torch.mean(torch.sum((reconst - inputs)**2, dim=[-1,-2,-3]))

            # (784, 400) smooth on weight
            smooth = self.reg(self.model.decoder.weight)

            # L1 sparse
            sparse = sparseness(code, order=1)
            loss = MSE + self.smooth_reg * smooth + self.sparse_reg * sparse

            SL += self.som_loss(self.model.decoder.weight.t().cpu().detach().numpy(), inputs.reshape(-1, 784).cpu().numpy())
            code_len = code.ge(0.01).sum(dim=1).float().mean()

            loss.backward()
            self.optimizer.step()

            MSE_loss += MSE.item()
            VQ_loss += VQ.item()
            Sparse_loss += sparse.item()
            Laplacian_loss += smooth.item()
            Code_len += code_len

        print('Train_MSE: %.6f | Train_VQ: %.6f | Train_Laplacian: %.6f | Train_Code_len: %.6f| Train_SOM: %.3f| Train_code_len: %.3f' % (MSE_loss/(batch_idx+1), VQ_loss/(batch_idx+1), Laplacian_loss/(batch_idx+1), Sparse_loss/(batch_idx+1), SL/(batch_idx+1), Code_len / (batch_idx+1)))
        self.logger.add(['Train_MSE', 'Train_VQ', 'Train_Laplacian', 'Train_Code_len', 'Train_SOM'], [MSE_loss/(batch_idx+1), VQ_loss/(batch_idx+1), Laplacian_loss/(batch_idx+1), Sparse_loss/(batch_idx+1), SL/(batch_idx+1)])
        self.scheduler.step()
        return loss

    def test(self):
        self.model.eval()
        MSE_loss = 0
        VQ_loss = 0
        Sparse_loss = 0
        Laplacian_loss = 0
        SL = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                reconst, code = self.model(inputs)
                MSE = self.criterion(reconst, inputs)

                VQ = torch.mean(torch.sum((reconst - inputs)**2, dim=[-1,-2,-3]))
                
                # (784, 400) smooth on weight
                smooth = self.reg(self.model.decoder.weight)

                sparse = sparseness(code, order=1)
                loss = MSE + self.smooth_reg * smooth + self.sparse_reg * sparse

                SL += self.som_loss(self.model.decoder.weight.t().cpu().detach().numpy(), inputs.reshape(-1, 784).cpu().numpy())

                MSE_loss += MSE.item()
                VQ_loss += VQ.item()
                Sparse_loss += sparse.item()
                Laplacian_loss += smooth.item()


        print('Test_MSE: %.6f | Test_VQ: %.6f | Test_Laplacian: %.6f | Test_Code_len: %.6f| Test_SOM: %.3f' % (MSE_loss/(batch_idx+1), VQ_loss/(batch_idx+1), Laplacian_loss/(batch_idx+1), Sparse_loss/(batch_idx+1), SL/(batch_idx+1)))
        self.logger.add(['Test_MSE', 'Test_VQ', 'Test_Laplacian', 'Test_Code_len', 'Test_SOM'], [MSE_loss/(batch_idx+1), VQ_loss/(batch_idx+1), Laplacian_loss/(batch_idx+1), Sparse_loss/(batch_idx+1), SL/(batch_idx+1)])
        return loss

    def save_model(self, check_path, round):
        print('Saving..')
        if not os.path.isdir(check_path):
            os.mkdir(check_path)
        torch.save(self.model.state_dict(), check_path+'/check_%d.pth'%round)

    def resume(self):
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(self.param.check_root), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(self.param.check_path)
        self.model.load_state_dict(checkpoint['net'])

    def vis_weight(self, vis_path, epoch):
        print('Visualizing Dictionary to {}'.format(vis_path))
        if not os.path.isdir(vis_path):
            os.mkdir(vis_path)

        images = self.model.decoder.weight.t().reshape(-1, 1, 28, 28).cpu().detach().numpy()
        
        # 100 * 28*28
        rows, cols = self.param.DIM_NUM, self.param.DIM_NUM
        image_height, image_width = self.param.h, self.param.w

        big_image = np.zeros((rows * image_height, cols * image_width))

        for idx, img in enumerate(images):
            row = idx // cols  
            col = idx % cols 
            big_image[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = img

        filepath = vis_path+'/epoch_{}.png'.format(epoch)
        imsave(filepath, big_image, cmap='gray')


