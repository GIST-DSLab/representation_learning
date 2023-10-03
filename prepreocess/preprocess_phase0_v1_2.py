import os
from glob import glob
import json
from utils import *
import pandas as pd
import re
from model import *

max_example_count = 10
max_len = 30
count = 0

SAMPLE_FLAG = False
SOFT_FLAG = False
CONCEPT_FLAG = True
V2_FLAG = False
MULTI = True

dataset_path = '../data'
train_dataset_path = 'phase0_train_v1_2'
valid_dataset_path = 'phase0_valid_v1_2'
original_path = '../original_data'

os.makedirs(f'{dataset_path}/{train_dataset_path}', exist_ok=True)
os.makedirs(f'{dataset_path}/{valid_dataset_path}', exist_ok=True)

# ARC의 최대 예제 개수는 10개 -> 794b24be.json
# Concept ARC의 최대 예제 개수는 5개

model = vae_v1_1(' ').to('cuda')
model.load_state_dict(torch.load('../result/Phase0_vae_v1_1_b64_lr1e-3_6.pt'))

train_path = f'{original_path}/training/*'
valid_path = f'{original_path}/evaluation/*'

train_files = glob(train_path)
valid_files = glob(valid_path)

train_dataset = {}
valid_dataset = {}
test_dataset = {}

train_tensor_path = []
valid_tensor_path = []

train_input = []
train_output = []
train_input_size = []
train_output_size = []
train_task_class = []

valid_input = []
valid_output = []
valid_input_size = []
valid_output_size = []
valid_task_class = []

padding_input = [[0 for _ in range(30)] for _ in range(30)]
padding_size = [-1, -1]


for i, file in enumerate(train_files):
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0]
    target_file_name = []

    train_task_input = []
    train_task_output = []
    train_task_input_size = []
    train_task_output_size = []
    for _ in range(max_example_count - len(train_data)):
        train_task_input.append(padding_input)
        train_task_output.append(padding_input)

        train_task_input_size.append(padding_size)
        train_task_output_size.append(padding_size)

    for i in range(len(train_data)):
        train_data[i]['input'] = (np.array(train_data[i]['input'])).tolist()
        train_input_y_len, train_input_x_len = np.array(train_data[i]['input']).shape
        train_output_y_len, train_output_x_len = np.array(train_data[i]['output']).shape
        input_size = (train_input_y_len, train_input_x_len)
        output_size = (train_output_y_len, train_output_x_len)

        # apply padding
        input_arr = [[0 if m > train_input_x_len-1 or n > train_input_y_len-1 else train_data[i]['input'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        output_arr = [[0 if m > train_output_x_len - 1 or n > train_output_y_len - 1 else train_data[i]['output'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        if True in [1 in input_arr[x] for x in range(30)] or True in [1 in output_arr[x] for x in range(30)]:
            pass
        if len(input_arr) < 30:
            a = []

        train_task_input.append(input_arr)
        train_task_output.append(output_arr)


    input_tensor = model.for_preprocessing(torch.tensor(train_task_input).to('cuda'))
    output_tensor = model.for_preprocessing(torch.tensor(train_task_output).to('cuda'))
    concat_feature = torch.cat((input_tensor.reshape(-1, 128), output_tensor.reshape(-1,128)),dim=1)
    torch.save(concat_feature, f'{dataset_path}/{train_dataset_path}/{file_name}.pt')

    train_input.append(train_task_input)
    train_output.append(train_task_output)
    train_tensor_path.append(f'{train_dataset_path}/{file_name}.pt')


for i, file in enumerate(valid_files):
    # print(file)
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0][:-1] if not CONCEPT_FLAG else file.split('\\')[-1].split('.')[0]
    target_file_name = []

    valid_task_input = []
    valid_task_output = []

    for _ in range(max_example_count - len(train_data)):
        valid_task_input.append(padding_input)
        valid_task_output.append(padding_input)

    for i in range(len(train_data)):
        train_data[i]['input'] = (np.array(train_data[i]['input'])).tolist()
        train_input_y_len, train_input_x_len = np.array(train_data[i]['input']).shape
        train_output_y_len, train_output_x_len = np.array(train_data[i]['output']).shape
        input_size = (train_input_y_len, train_input_x_len)
        output_size = (train_output_y_len, train_output_x_len)

        input_arr = [
            [0 if m > train_input_x_len - 1 or n > train_input_y_len - 1 else train_data[i]['input'][n][m] + 1 for m
             in range(max_len)] for n in range(max_len)]
        output_arr = [
            [0 if m > train_output_x_len - 1 or n > train_output_y_len - 1 else train_data[i]['output'][n][m] + 1
             for m in range(max_len)] for n in range(max_len)]
        if len(input_arr) < 30:
            a = []
        valid_task_input.append(input_arr)
        valid_task_output.append(output_arr)

    input_tensor = model.for_preprocessing(torch.tensor(valid_task_input).to('cuda'))
    output_tensor = model.for_preprocessing(torch.tensor(valid_task_output).to('cuda'))
    concat_feature = torch.cat((input_tensor.reshape(-1, 128), output_tensor.reshape(-1, 128)), dim=1)
    torch.save(concat_feature, f'{dataset_path}/{valid_dataset_path}/{file_name}.pt')

    valid_input.append(valid_task_input)
    valid_output.append(valid_task_output)
    valid_tensor_path.append(f'{valid_dataset_path}/{file_name}.pt')

train_json_data = {
    'input': train_input,
    'output': train_output,
    'tensor_path': train_tensor_path,
}

valid_json_data = {
    'input': valid_input,
    'output': valid_output,
    'tensor_path': valid_tensor_path,
}


with open(f'{dataset_path}/phase0_train_v1_2.json', 'w') as f:
    json.dump(train_json_data, f)

with open(f'{dataset_path}/phase0_valid_v1_2.json', 'w') as f:
    json.dump(valid_json_data, f)
