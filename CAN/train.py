import os
import time
import argparse
import random
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
import numpy as np

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.can import CAN
from training import train, eval

context.set_context(device_id=1, mode=1, device_target="Ascend")

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--check', action='store_true', help='测试代码选项')
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'

"""加载config文件"""
params = load_config(config_file)

"""设置随机种子"""
random.seed(params['seed'])
np.random.seed(params['seed'])
ms.set_seed(params['seed'])

if args.dataset == 'CROHME':
    train_loader, eval_loader = get_crohme_dataset(params)

model = CAN(params)
load_checkpoint(model, None, params['checkpoint'])
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

print(model.name)

import math
def update_lr(current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr   
    return new_lr

result = []
for epoch in range(240):
    for step in range(1104):
        result.append(update_lr(epoch, step, 1104, 240, 1))

optimizer = nn.Adadelta(model.get_parameters(), learning_rate =ms.Tensor(result), epsilon=float(params['eps']), weight_decay=float(params['weight_decay']))

if params['finetune']:
    print('加载预训练模型权重')
    print(f'预训练权重路径: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

"""在CROHME上训练"""
if args.dataset == 'CROHME':
    min_score, init_epoch = 0, 0

    for epoch in range(init_epoch, params['epochs'] // params['batch_size']):
        
        train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader)

        if epoch >= params['valid_start']:
            eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader)
            print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
            if eval_exprate > min_score and not args.check and epoch >= params['save_start']:
                min_score = eval_exprate
                save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
