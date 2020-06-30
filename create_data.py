import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle
from Configuration import Config

cfg = Config()

def create_data_DroneRF():
    dir_data = 'D:\\datasets\\DroneRF'
    list_samples = glob(pathname = os.path.join(dir_data, '**', '*_H*', '*.csv'), recursive = True)
    out_data = {0:[], 1:[], 2:[], 3:[]}
    bui = {'00000':{'type':0, 'info':'Background activities'},
           '10000':{'type':1, 'info':'Bebop drone, mode 1'},
           '10001':{'type':1, 'info':'Bebop drone, mode 2'},
           '10010':{'type':1, 'info':'Bebop drone, mode 3'},
           '10011':{'type':1, 'info':'Bebop drone, mode 4'},
           '10100':{'type':2, 'info':'AR drone, mode 1'},
           '10101':{'type':2, 'info':'AR drone, mode 2'},
           '10110':{'type':2, 'info':'AR drone, mode 3'},
           '10111':{'type':2, 'info':'AR drone, mode 4'},
           '11000':{'type':3, 'info':'Phantom drone, mode 1'}}
    _cnn_channel_part = int(2e7 // cfg.input_с)
    _cnn_part = int(_cnn_channel_part // cfg.input_l)
    _cnn_part_residual = int(_cnn_channel_part % cfg.input_l)
    for sample in tqdm(list_samples):
        sample_h = sample
        sample_l = sample.replace('H', 'L')
        with open(sample_h) as f_h:
            with open(sample_l) as f_l:
                csv_reader_h = csv.reader(f_h, delimiter=',')
                csv_reader_l = csv.reader(f_l, delimiter=',')
                sample_data = np.array(next(iter(csv_reader_l)) + next(iter(csv_reader_h))).astype(np.float)
        sample_data_out = []
        for sample_i in range(cfg.input_с):
            sample_data_ = sample_data[_cnn_channel_part * sample_i:_cnn_channel_part * (sample_i + 1)]
            sample_data_out.append(list(np.concatenate((np.mean(sample_data_[:_cnn_part + int(_cnn_part_residual // 2)].reshape((1,-1)), axis=1),
                                      np.mean(sample_data_[_cnn_part + int(_cnn_part_residual // 2):
                                      -(_cnn_part + int(_cnn_part_residual // 2))].reshape((cfg.input_l-2, _cnn_part)), axis=1),
                                      np.mean(sample_data_[-(_cnn_part + int(_cnn_part_residual // 2)) :].reshape((1,-1)), axis=1)))))
        out_data[bui[sample.split('\\')[-1][:5]]['type']].append(sample_data_out)
    with open(os.path.join(os.getcwd(), 'data', 'data.json'), "w") as write_file:
        json.dump(out_data, write_file)
    print('Complite')

def create_train_val_data():
    data_train = {}
    data_val = {}
    data_train_counter = 0
    data_val_counter = 0
    with open(os.path.join(os.getcwd(), 'data', 'data.json'), "r") as read_file:
        data = json.load(read_file)
    
    for class_i in tqdm(data.keys()):
        data_sample = data[class_i]
        for sample_i in range(len(data_sample)):
            if sample_i < len(data_sample) - 2:
                data_train[data_train_counter] = [data_sample[sample_i], 0 if class_i == '0' else 1]
                data_train_counter += 1
            else:
                data_val[data_val_counter] = [data_sample[sample_i], 0 if class_i == '0' else 1]
                data_val_counter += 1
    with open(os.path.join(os.getcwd(), 'data', 'train.json'), "w") as write_file:
        json.dump(data_train, write_file)
    with open(os.path.join(os.getcwd(), 'data', 'val.json'), "w") as write_file:
        json.dump(data_val, write_file)


if __name__ == '__main__':
    # create_data_DroneRF()
    # create_train_val_data()
    # with open(os.path.join(os.getcwd(), 'data', 'val.json'), "r") as f:
    #     data = json.load(f)
    # print()
    pass
