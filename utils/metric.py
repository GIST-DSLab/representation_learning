import torch
import torch.nn as nn
from sklearn import metrics

def emr(batch_size, class_num, output, target):
    temp_list = []
    categories = ['AboveBelow', 'Center', 'CleanUp', 'CompleteShape', 'Copy', 'Count', 'ExtendToBoundary', 'ExtractObjects', 'FilledNotFilled', 'HorizontalVertical', 'InsideOutside', 'MoveToBoundary', 'Order', 'SameDifferent', 'TopBottom2D', 'TopBottom3D']
    # categories = ['Move', 'Color', 'Object', 'Pattern', 'Count', 'Crop', 'Boundary', 'Center', 'Resize', 'Inside','Outside', 'Remove', 'Copy', 'Position', 'Direction', 'Bitwise', 'Connect', 'Order', 'Combine','Fill']
    for i in range(batch_size):
        is_correct = torch.where(torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).eq(target[i]).sum() == class_num,1, 0)
        if is_correct and batch_size == 1:
            temp_str = []
            for c in range(len(categories)):
                if target[i][c] == 1:
                    temp_str.append(categories[c])
            print(temp_str)
        temp_list.append(is_correct)
    return temp_list

def accuracy(batch_size, class_num, output, target):
    temp_list = []
    for i in range(batch_size):
        temp_list.append(torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).eq(target[i]).sum() / class_num)
    return temp_list

def precision(batch_size, class_num, output, target):
    temp_list = []
    for i in range(batch_size):
        temp_list.append(metrics.precision_score(target[i].detach().cpu().numpy() , torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).detach().cpu().numpy()))
    return temp_list

def recall(batch_size, class_num, output, target):
    temp_list = []
    for i in range(batch_size):
        temp_list.append(metrics.recall_score(target[i].detach().cpu().numpy() , torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).detach().cpu().numpy()))
    return temp_list

def f1(batch_size, class_num, output, target):
    temp_list = []
    for i in range(batch_size):
        temp_list.append(metrics.f1_score(target[i].detach().cpu().numpy() , torch.round(nn.functional.sigmoid(output.permute(1, 0, 2))[i]).detach().cpu().numpy()))
    return temp_list

def macro_average():
    pass

def micro_average():
    pass

def weighted_average():
    pass