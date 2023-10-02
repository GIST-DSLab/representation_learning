from glob import glob
import json
from utils import *
import pandas as pd
import re


max_example_count = 10
max_len = 30
count = 0

SAMPLE_FLAG = False
SOFT_FLAG = False
CONCEPT_FLAG = True
V2_FLAG = False
MULTI = True

dataset_path = '../data'
original_path = '../original_data'

# ARC의 최대 예제 개수는 10개 -> 794b24be.json
# Concept ARC의 최대 예제 개수는 5개


train_path = f'{original_path}/training/*'
valid_path = f'{original_path}/evaluation/*'

train_files = glob(train_path)
valid_files = glob(valid_path)

train_dataset = {}
valid_dataset = {}
test_dataset = {}

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
        train_task_input_size.append(input_size)
        train_task_output_size.append(output_size)

        # train_task_task_class.append(target_file_name)

    train_input.append(train_task_input)
    train_output.append(train_task_output)

    train_input_size.append(train_task_input_size)
    train_output_size.append(train_task_output_size)

    train_task_class.append(target_file_name)

    train_task_input = []
    train_task_output = []
    train_task_input_size = []
    train_task_output_size = []

    # for _ in range(max_example_count - len(valid_data)):
    #     train_task_input.append(padding_input)
    #     train_task_output.append(padding_input)
    #
    #     train_task_input_size.append(padding_size)
    #     train_task_output_size.append(padding_size)

    # for i in range(len(valid_data)):
    #     valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
    #     valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
    #     valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape
    #
    #     input_size = (valid_input_y_len, valid_input_x_len)
    #     output_size = (valid_output_y_len, valid_output_x_len)
    #
    #     # apply padding
    #     input_arr = [[0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
    #     output_arr = [ [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
    #
    #     if True in [11 in input_arr[x] for x in range(30)] or True in [11 in output_arr[x] for x in range(30)]:
    #         pass
    #     if len(input_arr) < 30:
    #         a = []
    #     train_task_input.append(input_arr)
    #     train_task_output.append(output_arr)
    #     train_task_input_size.append(input_size)
    #     train_task_output_size.append(output_size)
    #
    # train_input.append(train_task_input)
    # train_output.append(train_task_output)
    #
    # train_input_size.append(train_task_input_size)
    # train_output_size.append(train_task_output_size)

    if SAMPLE_FLAG:
        print(file)
        break


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
    valid_task_input_size = []
    valid_task_output_size = []

    for _ in range(max_example_count - len(train_data)):
        valid_task_input.append(padding_input)
        valid_task_output.append(padding_input)

        valid_task_input_size.append(padding_size)
        valid_task_output_size.append(padding_size)

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

        valid_task_input_size.append(input_size)
        valid_task_output_size.append(output_size)

    valid_input.append(valid_task_input)
    valid_output.append(valid_task_output)

    valid_input_size.append(valid_task_input_size)
    valid_output_size.append(valid_task_output_size)

    # valid_task_input = []
    # valid_task_output = []
    # valid_task_input_size = []
    # valid_task_output_size = []
    #
    # for _ in range(max_example_count - len(valid_data)):
    #     valid_task_input.append(padding_input)
    #     valid_task_output.append(padding_input)
    #
    #     valid_task_input_size.append(padding_size)
    #     valid_task_output_size.append(padding_size)

    # for i in range(len(valid_data)):
    #     valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
    #     valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
    #     valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape
    #
    #     input_size = (valid_input_y_len, valid_input_x_len)
    #     output_size = (valid_output_y_len, valid_output_x_len)
    #
    #     input_arr = [
    #         [0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m] + 1 for m
    #          in range(max_len)] for n in range(max_len)]
    #     output_arr = [
    #         [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m] + 1
    #          for m in range(max_len)] for n in range(max_len)]
    #
    #     if len(input_arr) < 30:
    #         a = []
    #
    #     valid_task_input.append(input_arr)
    #     valid_task_output.append(output_arr)
    #
    #     valid_task_input_size.append(input_size)
    #     valid_task_output_size.append(output_size)
    #
    # valid_input.append(valid_task_input)
    # valid_output.append(valid_task_output)
    #
    # valid_input_size.append(valid_task_input_size)
    # valid_output_size.append(valid_task_output_size)

train_json_data = {
    'input': train_input,
    'output': train_output,
    'input_size': train_input_size,
    'output_size': train_output_size,
}

valid_json_data = {
    'input': valid_input,
    'output': valid_output,
    'input_size': valid_input_size,
    'output_size': valid_output_size,
}


with open(f'{dataset_path}/phase0_train_v2.json', 'w') as f:
    json.dump(train_json_data, f)

with open(f'{dataset_path}/phase0_valid_v2.json', 'w') as f:
    json.dump(valid_json_data, f)
