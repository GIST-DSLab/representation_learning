import random

from torch.utils.data import Dataset
import numpy as np
import json
import torch
from collections import OrderedDict

class ARCDataset(Dataset):
  def __init__(self, file_name, mode=None, permute_mode=False, rotate_mode=True):
    self.dataset = None
    self.mode = mode
    self.permute_mode = permute_mode
    self.rotate_mode = rotate_mode
    # self.permute_color = np.random.choice(11, 11, replace=False)
    with open(file_name, 'r') as f:
      self.dataset = json.load(f)

  def __len__(self):
    if self.mode == 'phase0_v1':
        return len(self.dataset['data'])
    else:
        return len(self.dataset['input'])

  def __getitem__(self,idx):
    if self.mode == 'phase0_v1':
        x = self.dataset['data'][idx]
        size = self.dataset['size'][idx]

        if self.rotate_mode:
            n = random.randint(0, 4)
            x = np.rot90(x, n).tolist()

        if self.permute_mode:
            self.permute_color = [0] + np.random.choice([i for i in range(1,11)], 10, replace=False).tolist()
            for i in range(30):
                for j in range(30):
                    x[i][j] = self.permute_color[x[i][j]]
        return torch.tensor(x), torch.tensor(size)
    elif self.mode == 'phase0_v2':
        x = self.dataset['input'][idx]
        y = self.dataset['output'][idx]

        if self.rotate_mode:
            n = random.randint(0, 4)
            for exam in range(len(x)):
                x[exam] = np.rot90(x[exam], n).tolist()
                y[exam] = np.rot90(y[exam], n).tolist()


        self.permute_color = [0] + np.random.choice([i for i in range(1, 11)], 10, replace=False).tolist()
        if 'v2' in self.mode:
            for exam in range(len(x)):
                for i in range(len(x[1])):
                    for j in range(len(x[2])):
                        x[exam][i][j] = self.permute_color[x[exam][i][j]]
                for i in range(len(y[1])):
                    for j in range(len(y[2])):
                        y[exam][i][j] = self.permute_color[y[exam][i][j]]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class LARC_Dataset(Dataset):
  def __init__(self, grid_files, LARC_file_name):
    self.grid_files = grid_files
    self.LARC_dataset = None
    with open(LARC_file_name, 'r') as f:
        self.LARC_dataset = json.load(f)

  def __len__(self):
    return len(self.LARC_dataset['task_name'])


  def __getitem__(self,idx):
    grid_file = self.grid_files[idx]
    task_name = self.LARC_dataset['task_name'][idx]
    task_description_output = self.LARC_dataset['description_output'][idx]
    return grid_file, task_name, task_description_output

class New_ARCDataset(Dataset):
    def __init__(self, file_name, mode=None, permute_mode=False, rotate_mode=False):
        self.dataset = None
        self.mode = mode
        self.count_boundary = 2500
        self.count = 0
        self.permute_mode = permute_mode
        self.rotate_mode = rotate_mode
        self.permute_color = np.random.choice(11, 11, replace=False)
        with open(file_name, 'r') as f:
            self.dataset = json.load(f)
        if self.mode == 'task':
            task_list = list(set(self.dataset['task']))
            self.task_dict = OrderedDict()
            for i, task in enumerate(task_list):
                self.task_dict[task] = i
        elif 'concept' in mode and 'multi' in mode:
            self.categories = ['AboveBelow', 'Center', 'CleanUp', 'CompleteShape', 'Copy', 'Count', 'ExtendToBoundary', 'ExtractObjects', 'FilledNotFilled', 'HorizontalVertical', 'InsideOutside', 'MoveToBoundary', 'Order', 'SameDifferent', 'TopBottom2D', 'TopBottom3D']
        elif 'multi' in mode:
            self.categories = ['Move', 'Color', 'Object', 'Pattern', 'Count', 'Crop', 'Boundary', 'Center', 'Resize', 'Inside', 'Outside', 'Remove', 'Copy', 'Position', 'Direction', 'Bitwise', 'Connect', 'Order', 'Combine', 'Fill']

    def __len__(self):
        if self.mode == 'Auto_encoder':
                return len(self.dataset['data'])
        else:
            return len(self.dataset['input'])

    def __getitem__(self,idx):
        if self.mode == 'Auto_encoder':
            x = self.dataset['data'][idx]
            size = self.dataset['size'][idx]
            if self.permute_mode:
                self.permute_color = [0] + np.random.choice([i for i in range(1,11)], 10, replace=False).tolist()
                for i in range(30):
                    for j in range(30):
                        x[i][j] = self.permute_color[x[i][j]]
            return torch.tensor(x), torch.tensor(size)
        elif 'multi-bc' in self.mode:
            x = self.dataset['input'][idx]
            y = self.dataset['output'][idx]
            multi_labels = []

            if self.rotate_mode:
                n = random.randint(0,4)
                if 'v2' in self.mode:
                    for exam in range(len(x)):
                        x[exam] = np.rot90(x[exam], n).tolist()
                        y[exam] = np.rot90(y[exam], n).tolist()
                else:
                    x = np.rot90(x, n).tolist()
                    y = np.rot90(y, n).tolist()

            if self.permute_mode and self.count % self.count_boundary == 0:
                if self.count_boundary > 1:
                    self.count_boundary -= 1
                self.permute_color = [0] + np.random.choice([i for i in range(1,11)], 10, replace=False).tolist()
                if 'v2' in self.mode:
                    for exam in range(len(x)):
                        for i in range(len(x[1])):
                            for j in range(len(x[2])):
                                x[exam][i][j] = self.permute_color[x[exam][i][j]]
                        for i in range(len(y[1])):
                            for j in range(len(y[2])):
                                y[exam][i][j] = self.permute_color[y[exam][i][j]]
                else:
                    for i in range(len(x[1])):
                        for j in range(len(x[2])):
                            x[i][j] = self.permute_color[x[i][j]]
                    for i in range(len(y[1])):
                        for j in range(len(y[2])):
                            y[i][j] = self.permute_color[y[i][j]]

            for category in self.categories:
                temp = [1 if category in self.dataset['task'][idx] else 0]
                multi_labels.append(temp)
            self.count += 1
            return torch.tensor(x), torch.tensor(y), torch.tensor(multi_labels)
        elif 'multi-soft' in self.mode:
            x = self.dataset['input'][idx]
            y = self.dataset['output'][idx]
            if self.permute_mode:
                self.permute_color = np.random.choice(11, 11, replace=False)
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        x[i][j] = self.permute_color[x[i][j]]
                        y[i][j] = self.permute_color[y[i][j]]
            multi_labels = [1 if category in self.dataset['task'][idx] else 0 for category in self.categories]
            return torch.tensor(x), torch.tensor(y), torch.tensor(multi_labels)
        else:
            x = self.dataset['input'][idx]
            y = self.dataset['output'][idx]
            if self.permute_mode:
                self.permute_color = [0] + np.random.choice([i for i in range(1,11)], 10, replace=False).tolist()
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        x[i][j] = self.permute_color[x[i][j]]
                        y[i][j] = self.permute_color[y[i][j]]
            task = self.task_dict[self.dataset['task'][idx]]
            return torch.tensor(x), torch.tensor(y), task