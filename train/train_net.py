# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import sys
import math
import torch
import shutil
import logging
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from torchvision import transforms
from time import localtime, strftime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.raster_label_visualizer import RasterLabelVisualizer
import sys
from models.SiamNet import SiamNet
from utils.train_utils import AverageMeter
from utils.train_utils import load_json_files, dump_json_files
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset_shard_load import DisasterDataset

config = {'labels_dmg': [0, 1, 2, 3, 4],
          'labels_bld': [0, 1],
          'weights_seg': [1, 15],
          'weights_damage': [1, 35, 70, 150, 120],
          'weights_loss': [0.5, 0.5, 0],
          'mode': 'bld',
          'init_learning_rate': 0.0005,#dmg: 0.005, #UNet: 0.01,           
          'device': 'cuda',
          'epochs': 2,
          'batch_size': 5,
          'num_chips_to_viz': 1,
          'experiment_name': 'train_UNet', #train_dmg
          'out_dir': './outputs/',
          'data_dir_shards': './xBD_sliced_augmented_20_alldisasters_final_mdl_npy',
          'shard_no': 0,
          'disaster_splits_json': './constants/splits/final_mdl_all_disaster_splits_sliced_img_augmented_20.json',
          'disaster_mean_stddev': './constants/splits/all_disaster_mean_stddev_tiles_0_1.json',
          'label_map_json': './constants/class_lists/xBD_label_map.json',
          'starting_checkpoint_path': './outputs/UNet_all_data_dmg/checkpoints/checkpoint_epoch120_2021-06-30-10-28-49.pth.tar'}


logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='{asctime} {levelname} {message}',
                    style='{',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Using PyTorch version {torch.__version__}.')
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}.')


def main():

    global viz, labels_set_dmg, labels_set_bld
    global xBD_train, xBD_val
    global train_loader, val_loader, test_loader
    global weights_loss, mode

    xBD_train, xBD_val = load_dataset()

    train_loader = DataLoader(xBD_train, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(xBD_val, batch_size=config['batch_size'], shuffle=False, num_workers=1, pin_memory=False)

    # label_map = load_json_files(config['label_map_json'])
    # viz = RasterLabelVisualizer(label_map=label_map)

    # labels_set_dmg = config['labels_dmg']
    # labels_set_bld = config['labels_bld']
    # mode = config['mode']

    # eval_results_tr_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    # eval_results_tr_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    # eval_results_val_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    # eval_results_val_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])

    model = SiamNet().to(device=device)
    model_summary(model)
    model.train()
    for batch_idx, data in enumerate(train_loader): 
                         
        x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
        x_post = data['post_image'].to(device=device)  
        scores = model(x_pre, x_post)
  

def load_dataset():
    splits = load_json_files(config['disaster_splits_json'])
    data_mean_stddev = load_json_files(config['disaster_mean_stddev'])

    train_ls = [] 
    val_ls = []
    for item, val in splits.items():
        train_ls += val['train'] 
        val_ls += val['val']
    xBD_train = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'train', data_mean_stddev, transform=True, normalize=True)
    xBD_val = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'val', data_mean_stddev, transform=False, normalize=True)

    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

    return xBD_train, xBD_val

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
        bias = (i.bias is not None)
    except:
        bias = False  
    if not bias:
        param =model_parameters[j].numel()+model_parameters[j+1].numel()
        j = j+2
    else:
        param =model_parameters[j].numel()
        j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       


if __name__ == "__main__":
    main()