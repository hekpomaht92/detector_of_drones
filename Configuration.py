import os
import json
import torch
import numpy as np
from tqdm import tqdm

class Config:
    
    def __init__(self):
        self.check_folders()
        self.get_files()
        self.get_mean_std_values()
        self.get_weights_coef_values()
        self.get_max_value()

        self.input_с = 20
        self.input_l = 2048
        self.class_n = 2
        self.batch_size = 220
        self.num_epoch = 5000

        self.pretrained_weights = None
        self.device = "cuda:0"

        self.learning_rate = 1e-4
        self.drop_rate = 0.5

    def check_folders(self):
        if not os.path.exists(os.path.join(os.getcwd(), 'weights')):
            os.makedirs(os.path.join(os.getcwd(), 'weights'))
        if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
            os.makedirs(os.path.join(os.getcwd(), 'logs'))
        if not os.path.exists(os.path.join(os.getcwd(), 'data')):
            os.makedirs(os.path.join(os.getcwd(), 'data'))

    def get_files(self):
        try:
            self.dir_train = os.path.join(os.getcwd(), 'data', 'train.json')
            self.dir_val = os.path.join(os.getcwd(), 'data', 'val.json')
            self.num_data = {}
            with open(self.dir_train, "r") as f:
                    self.num_data['train'] = len(json.load(f))
            with open(self.dir_val, "r") as f:
                self.num_data['val'] = len(json.load(f))
        except:
            print('Train and val files issue')
    
    def get_max_value(self):
        try:
            with open(os.path.join(os.getcwd(), 'data', 'max_value.txt'), "r") as f:
                line = f.readline()
            self.max_value = float(line.strip())
        except:
            print('Max value issue')

    def get_mean_std_values(self):
        try:
            with open(os.path.join(os.getcwd(), 'data', 'mean_std_values.txt'), "r") as f:
                lines = f.readlines()
            self.mean_value = np.array(list(map(float, (lines[0].split()))))
            self.std_value = np.array(list(map(float, (lines[1].split()))))
        except:
            print('Mean and std file issue')
    
    def get_weights_coef_values(self):
        try:
            with open(os.path.join(os.getcwd(), 'data', 'weights_coefficients.txt'), "r") as f:
                line = f.readline()
            self.weight_coefficients = np.array(list(map(float, (line.split()))))
        except:
            print('Weights coefficient file issue')

    def computing_max_value(self):
        max_value = 0.0

        with open(os.path.join(os.getcwd(), 'data', 'train.json'), "r") as f:
            data = json.load(f)

        for i in tqdm(data.keys()):
            x_data = np.array(data[i][0])
            max_value_ = np.amax(x_data)
            if max_value_ > max_value:
                max_value = max_value_

        with open(os.path.join(os.getcwd(), 'data', 'max_value.txt'), 'w') as f:
            f.write(str(max_value))

    def computing_mean_std(self):
        _mean_absolut = np.array(list(map(float, [0 for i in range(self.input_l)])))
        _std_absolut = np.array(list(map(float, [0 for i in range(self.input_l)])))
        _len_absolut = self.num_data['train']

        with open(self.dir_train, "r") as f:
            data = json.load(f)

        for i in tqdm(data.keys()):
            x_data = np.array(data[i][0]).astype("float32")
            x_data /= self.max_value
            means = x_data.mean(dtype='float32')
            stds = x_data.std(dtype='float32')

            _mean_absolut += means
            _std_absolut += stds

        _mean_absolut /= _len_absolut
        _std_absolut /= _len_absolut

        with open(os.path.join(os.getcwd(), 'data', 'mean_std_values.txt'), 'w') as f:
            _mean_absolut = ' '.join(map(str, list(_mean_absolut)))
            _std_absolut = ' '.join(map(str, list(_std_absolut)))
            f.write('{}\n{}'.format(_mean_absolut, _std_absolut))
    
    def computing_class_weights(self):
        num_labels = [0 for i in range(self.class_n)]
        with open(self.dir_train, "r") as f:
            data = json.load(f)
        
        for i in tqdm(data.keys()):
            y_data = int(data[i][1])
            num_labels[y_data] += 1
        
        num_labels = self.num_data['train'] / np.array(num_labels)
        
        with open(os.path.join(os.getcwd(), 'data', 'weights_coefficients.txt'), 'w') as f:
            num_labels = ' '.join(map(str, list(num_labels)))
            f.write('{}'.format(num_labels))
        
if __name__ == "__main__":
    cfg = Config()
    # print()
    # cfg.computing_max_value()
    # cfg.computing_mean_std()
    # cfg.computing_class_weights()