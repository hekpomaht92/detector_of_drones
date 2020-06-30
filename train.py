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
cfg = Config()


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
        y_data = torch.tensor(self.data[str(idx)][1])
        sample = {'data': x_data, 'label': y_data}
        return sample


class Trainer:

    def __init__(self):
        self.model = generate_model()
        self.model.to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        # self.criterion = nn.CrossEntropyLoss().to(cfg.device)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cfg.weight_coefficients)).to(cfg.device)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.writer = SummaryWriter('logs/{}'.format(time.time()))
        self.dataloaders = {'train': torch.utils.data.DataLoader(
                            DataGenerator(input_path=cfg.dir_train, num_data=cfg.num_data['train']),
                            batch_size=cfg.batch_size, shuffle=True),
                            'val': torch.utils.data.DataLoader(
                            DataGenerator(input_path=cfg.dir_val, num_data=cfg.num_data['val']),
                            batch_size=8, shuffle=True)}

    def write_logs(self, label, pred, i, epoch, phase, loss):
        y_true = torch.squeeze(label).to('cpu').numpy().reshape(-1)
        y_pred = torch.squeeze(pred).to('cpu').numpy().reshape(-1)
        out_metrics = metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)
        print("Phase: {}, Epoch: {:05}, Iter: {:02}, Loss: {:.5f}, Accuracy: {:.3f}"
              .format(phase, epoch, i, loss, out_metrics['accuracy']))
        writer_step = epoch
        # writer_step = (epoch - 1) * (cfg.num_data[phase] // 10) // cfg.batch_size + i // 10 + 1
        # if i % 10 == 0:
        self.writer.add_scalar('{} loss'.format(phase),
                                loss,
                                writer_step)
        self.writer.add_scalar('{} accuracy'.format(phase),
                                out_metrics['accuracy'],
                                writer_step)
        self.writer.add_scalar('{} avg precision'.format(phase),
                                out_metrics['macro avg']['precision'],
                                writer_step)
        self.writer.add_scalar('{} avg recall'.format(phase),
                                out_metrics['macro avg']['recall'],
                                writer_step)
        self.writer.add_scalar('{} avg f1-score'.format(phase),
                                out_metrics['macro avg']['f1-score'],
                                writer_step)

        for i_class in range(cfg.class_n):
            if str(i_class) in out_metrics.keys():
                self.writer.add_scalar('{} {} precision'.format(phase, i_class),
                                        out_metrics[str(i_class)]['precision'],
                                        writer_step)
                self.writer.add_scalar('{} {} recall'.format(phase, i_class),
                                        out_metrics[str(i_class)]['recall'],
                                        writer_step)
                self.writer.add_scalar('{} {} f1-score'.format(phase, i_class),
                                        out_metrics[str(i_class)]['f1-score'],
                                        writer_step)
    
    def train_epoch(self, epoch, phase='train'):
        self.model.train()
        for i, sample in enumerate(self.dataloaders[phase], 0):
            
            inputs = sample['data'].to(cfg.device)
            label = sample['label'].type(torch.LongTensor).to(cfg.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = outputs.max(1, keepdim=True)[1]
            
            self.write_logs(label=label, pred=pred, i=i, epoch=epoch, phase=phase, loss=loss.item())

        torch.save(self.model.state_dict(), os.path.join('weights', 'epoch_{:02}.pt'.format(epoch)))

    def validation_epoch(self, epoch, phase='val'):
        self.model.eval()
        for i, sample in enumerate(self.dataloaders[phase], 0):
            
            inputs = sample['data'].to(cfg.device)
            label = sample['label'].type(torch.LongTensor).to(cfg.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label)
                pred = outputs.max(1, keepdim=True)[1]

            self.write_logs(label=label, pred=pred, i=i, epoch=epoch, phase=phase, loss=loss.item())

    def train_model(self):
        for epoch in range(1, cfg.num_epoch + 1):

            # if epoch < cfg.initial_epoch:
            #     self.scheduler.step(epoch)
            #     continue
            
            print('Epoch {}/{}'.format(epoch, cfg.num_epoch - 1))
            print('-' * 10)

            self.train_epoch(epoch)
            self.validation_epoch(epoch)
            # self.scheduler.step()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model()