import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import Configuration

cfg = Configuration.Config()

def generate_model(pretrained_weights=cfg.pretrained_weights):
    model = My_Model()
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))
    return model


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(4, 2, 1),
            nn.ReLU(),
            nn.Dropout(cfg.drop_rate)
        )
    def forward(self, x):
        return self.block(x)


class Output(nn.Module):
    def __init__(self, in_channels, h_channels, bias=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.Linear(in_channels, h_channels, bias=bias),
            nn.BatchNorm1d(h_channels),
            nn.ReLU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(h_channels, cfg.class_n, bias=bias),
        )
    def forward(self, x):
        return self.block(x)


class My_Model(nn.Module):

    def __init__(self):
        super(My_Model, self).__init__()
        self.block = torch.nn.Sequential(
            CNN(cfg.input_с, 64),  # 2048 -> 512
            CNN(64, 128),          # 512  -> 128 
            CNN(128, 256),         # 128  -> 32
            CNN(256, 256)          # 32   -> 8
        )
        self.out = Output(2048, 512)

    def forward(self, x):
        x = self.block(x)
        x = x.view((x.shape[0], -1))
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = generate_model(None)
    # torch.save(model, 'model.pt')
    # model.cuda()
    # example = torch.rand(cfg.batch_size, cfg.input_с, cfg.input_l).cuda()
    # torch.onnx.export(model, example, 'model.onnx')
    # out = model(example)
    # print(out)
    # print("Complite")
    
