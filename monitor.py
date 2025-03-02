import os
import glob
import shutil
import time
from auto_encoder import AE
from hyper_param import mnist_hyper_param
from utils.logger import logger


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)

  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    os.mkdir(os.path.join(path, 'check'))
    os.mkdir(os.path.join(path, 'vis'))
    os.mkdir(os.path.join(path, 'feature'))
    os.mkdir(os.path.join(path, 'log'))
    
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


class Monitor():
    def __init__(self, param):
        self.log = logger()
        self.param = param
        self.auto_encoder = AE(self.param, self.log)

        self.path = '{}-{}-{}'.format(self.param.data_name, self.param.reg, self.param.reg_s)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_path = self.path + '/' + time.strftime("%Y%m%d-%H%M%S")

        create_exp_dir(self.save_path, scripts_to_save=glob.glob('*.py'))

    def one_run(self, round):
        for epoch in range(self.param.total_epoch):
            print("Epoch:", epoch)
            self.log.add(['Epoch'], [epoch])
            loss = self.auto_encoder.train()
            loss = self.auto_encoder.test()

            self.auto_encoder.vis_weight(os.path.join(self.save_path, 'vis', str(round)), epoch)
        self.auto_encoder.save_model(os.path.join(self.save_path, 'check'), round)
        self.log.write(os.path.join(self.save_path, 'log'), round)

    def reset_auto_encoder(self):
        self.log.reset()
        self.auto_encoder = AE(self.param, self.log)
        

    def train_model(self):
        for i in range(self.param.round):
            print('[Round %d]'%i)
            self.one_run(i)
            self.reset_auto_encoder()


if __name__ == "__main__":
    monitor = Monitor(mnist_hyper_param())
    monitor.train_model()