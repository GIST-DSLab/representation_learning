from lion_pytorch import Lion
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import wandb
import time
from time import localtime
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def seed_fix(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def set_wandb(seed, mode, kind_of_loss, lr, epochs, train_batch_size, valid_batch_size, model_name, use_permute, optimizer, use_pretrain, dataset_name='ARC'):
    tm = localtime(time.time())
    run = wandb.init(project=f'{mode}_{dataset_name}', entity='whatchang', )
    config = {
        'seed': seed,
        'learning_rate': lr,
        'epochs': epochs,
        'train_batch_size': train_batch_size,
        'valid_batch_size': valid_batch_size,
        'use_permute': use_permute,
        'kind_of_loss': kind_of_loss,
        'optimizer': optimizer,
        'pretrain': use_pretrain,
        'model_name': model_name,
        'time': f'd{tm.tm_mday}_h{tm.tm_hour}_m{tm.tm_min}_s{tm.tm_sec}'
    }
    wandb.config.update(config)
    wandb.run.name =f'{model_name}_e{epochs}'
    wandb.run.save()
    return run

def set_loss(loss_name):
    if loss_name == 'mse':
        loss = nn.MSELoss()
    elif loss_name == 'cross':
        loss = nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        loss = nn.BCELoss()

    return loss

def set_optimizer(optimizer_name, model, lr):
    if optimizer_name== 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'lion':
        optimizer = Lion(model.parameters(), lr=lr)

    return optimizer

def set_lr_scheduler(optimizer, scheduler_name, lr_lambda, step_size, gamma):
    if scheduler_name == 'lambdalr':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: lr_lambda ** epoch)
    elif scheduler_name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    return scheduler

def model_freeze(self, model):
    for param in model.parameters():
        param.requires_grad = False

def recursive_exchange(data, color_index, current_index, current_index_list):
    target_index = color_index.tolist().index(current_index)
    if target_index not in current_index_list:
        current_index_list.append(target_index)
        data= recursive_exchange(data, color_index, target_index, current_index_list)
    else:
        cycle_flag = len(current_index_list) < 10
        if cycle_flag:
            for i in range(11):
                if i not in current_index_list:
                    current_index = i
            target_index = color_index.tolist().index(current_index)
            current_index_list.append(target_index)
            data = recursive_exchange(data, color_index, target_index, current_index_list)
        else:
            data = torch.where(data == target_index, -1, data)
            return data
    data = torch.where(data == current_index, target_index, data)
    return data

def check_output(model, mode):
    with open(f'./data/easy_{mode}.json', 'r') as f:
        other_example = json.load(f)
    other_input = torch.tensor(other_example['input'], dtype=torch.float32).to('cuda')
    other_output = torch.tensor(other_example['output'], dtype=torch.float32).to('cuda')
    other_input_size = other_example['input_size']
    other_output_size = other_example['output_size']

    model.action_vector = nn.Parameter(torch.load(f'./experiment_feature/{mode}_action_pre.pt')['action_vector'])

    output = model(other_input)
    output = output.squeeze().permute(1, 2, 0)
    output = model.proj(output)
    round_output = torch.argmax(output, dim=2)
    return round_output, other_input, other_output, other_input_size, other_output_size

