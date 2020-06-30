import numpy as np
import time
from multiprocessing import set_start_method
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import metrics
from multiprocessing import set_start_method
from model import generate_model
from Configuration import Config

np.random.seed(42)
set_start_method('spawn', True)

class ConfigTest(Config):

    def __init__(self):
        Config.__init__(self)
        self.get_test_file()
        self.batch_size = 1
        self.pretrained_weights = os.path.join(os.getcwd(), 'weights', 'epoch_4939.pt')

    def get_test_file(self):
        try:
            self.dir_test = os.path.join(os.getcwd(), 'data', 'test.json')
            with open(self.dir_test, "r") as f:
                    self.num_data['test'] = len(json.load(f))
        except:
            print('test file issue')

class DataGenerator(Dataset):

    def __init__(self, input_path, num_data):
        with open(input_path, "r") as f:
            self.data = json.load(f)
        self.num_data = num_data
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        x_data = np.array(self.data[str(idx)][0]).astype("float32")
        x_data /= cfg.max_value
        x_data = (x_data - cfg.mean_value) / cfg.std_value
        x_data = torch.Tensor(x_data)
        sample = {'data': x_data}
        return sample


cfg = ConfigTest()


class Tester:

    def __init__(self):
        self.model = generate_model(cfg.pretrained_weights)
        self.model.to(cfg.device)
        self.dataloader = torch.utils.data.DataLoader(
                          DataGenerator(input_path=cfg.dir_test, num_data=cfg.num_data['test']),
                          batch_size=cfg.batch_size)
    
    def test_sample(self):
        self.model.eval()
        for i, sample in enumerate(self.dataloader):
            inputs = sample['data'].to(cfg.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                pred = np.squeeze(outputs.max(1, keepdim=True)[1].to('cpu').numpy())
            print('Sample #{}: '.format(i) + str(pred))


if __name__ == '__main__':
    tester = Tester()
    tester.test_sample()
            
