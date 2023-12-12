from glob import glob
import json
from utils import *
import pandas as pd
import re

max_example_count = 5
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

if not CONCEPT_FLAG:
    if MULTI:
        arc_task_sample = f'{original_path}/wongyu_arc50.csv'
        df_arc_task_sample = pd.read_csv(arc_task_sample, encoding='utf-8')

    train_path = f'{original_path}/training/*'
    test_path = f'{original_path}/evaluation/*'
else:
    arc_task_sample_train = f'{original_path}/concept_ARC_sample2_train.csv'
    arc_task_sample_test = f'{original_path}/concept_ARC_sample2_test.csv'
    df_arc_task_sample_train = pd.read_csv(arc_task_sample_train, encoding='utf-8')
    df_arc_task_sample_test = pd.read_csv(arc_task_sample_test, encoding='utf-8')

    train_path = f'{original_path}/concept_train/*'
    test_path = f'{original_path}/concept_eval/*'

train_files = glob(train_path)
test_files = glob(test_path)

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

test_input = []
test_output = []
test_input_size = []
test_output_size = []
test_task_class = []

padding_input = [[0 for _ in range(30)] for _ in range(30)]
padding_size = [-1, -1]
padding_task = "TopBottom"


for i, file in enumerate(train_files):
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    # if max_example_count < len(train_data):
    #     print(max_example_count, file)
    # max_example_count = max_example_count if max_example_count >= len(train_data) else len(train_data)
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0]
    target_file_name = []
    if MULTI:
        if CONCEPT_FLAG:
            if not V2_FLAG:
                a = re.search(f'[A-Za-z]+', df_arc_task_sample_train['Name'][i])
                if a[0] == "TopBottom":
                    b = a.string[:-1] if '10' not in a.string else a.string[:-2]
                else:
                    b = a[0]
                target_file_name.append(b)
                try:
                    temp_list = df_arc_task_sample_train[df_arc_task_sample_train['Name'] == file_name]['tag'].tolist()[0].split(', ')
                except:
                    print(1)
                for temp_tag in temp_list:
                    target_file_name.append(temp_tag)
            else:
                temp_list = df_arc_task_sample_train[df_arc_task_sample_train['Name'] == file_name]['class'].tolist()[0].split(', ')
                for temp_tag in temp_list:
                    target_file_name.append(temp_tag)
        else:
            target_file_name = df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['class'].tolist()
            if not target_file_name:
                continue
            target_file_name = df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['class'].tolist()[0]
        padding_task = target_file_name

    train_task_input = []
    train_task_output = []
    train_task_input_size = []
    train_task_output_size = []
    train_task_task_class = []
    for _ in range(max_example_count - len(train_data)):
        train_task_input.append(padding_input)
        train_task_output.append(padding_input)

        train_task_input_size.append(padding_size)
        train_task_output_size.append(padding_size)

        # train_task_task_class.append(padding_task)

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

    valid_task_input = []
    valid_task_output = []
    valid_task_input_size = []
    valid_task_output_size = []
    valid_task_task_class = []
    for i in range(len(valid_data)):
        valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
        valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
        valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape

        input_size = (valid_input_y_len, valid_input_x_len)
        output_size = (valid_output_y_len, valid_output_x_len)

        # apply padding
        input_arr = [[0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m]+1 for m in range(max_len)] for n in range(max_len)]
        output_arr = [ [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m]+1 for m in range(max_len)] for n in range(max_len)]

        if True in [11 in input_arr[x] for x in range(30)] or True in [11 in output_arr[x] for x in range(30)]:
            pass

        # train_input.append(input_arr)
        # train_output.append(output_arr)
        #
        # train_input.append(valid_data[i]['input'])
        # train_output.append(valid_data[i]['output'])
        #
        # train_input_size.append(input_size)
        # train_output_size.append(output_size)
        #
        # train_task_class.append(target_file_name)

        # valid_input.append(input_arr)
        # valid_output.append(output_arr)

        # valid_input.append(valid_data[i]['input'])
        # valid_output.append(valid_data[i]['output'])
        #
        # valid_input_size.append(input_size)
        # valid_output_size.append(output_size)
        #
        # valid_task_class.append(target_file_name)

    # count += 1
    # if count == 10:
    #     break
    if SAMPLE_FLAG:
        print(file)
        break


for i, file in enumerate(test_files):
    # print(file)
    with open(file, 'rb') as f:
        data = json.load(f)
    train_data = data['train']
    valid_data = data['test']
    file_name = file.split('\\')[-1].split('.')[0][:-1] if not CONCEPT_FLAG else file.split('\\')[-1].split('.')[0]
    target_file_name = []

    test_task_input = []
    test_task_output = []
    test_task_input_size = []
    test_task_output_size = []
    test_task_task_class = []

    for _ in range(max_example_count - len(train_data)):
        test_task_input.append(padding_input)
        test_task_output.append(padding_input)

        test_task_input_size.append(padding_size)
        test_task_output_size.append(padding_size)

        # test_task_task_class.append(padding_task)
    # if max_example_count > len(train_data):
    #     print(max_example_count, file)

    # max_example_count = max_example_count if max_example_count >= len(train_data) else len(train_data)
    if MULTI:
        if CONCEPT_FLAG:
            if not V2_FLAG:
                a = re.search(f'[A-Za-z]+', df_arc_task_sample_test['Name'][i])
                if a[0] == "TopBottom":
                    b = a.string[:-1] if '10' not in a.string else a.string[:-2]
                else:
                    b = a[0]
                target_file_name.append(b)
                try:
                    temp_list = df_arc_task_sample_test[df_arc_task_sample_test['Name'] == file_name]['tag'].tolist()[0].split(', ')
                except:
                    print(1)
                for temp_tag in temp_list:
                    target_file_name.append(temp_tag)
            else:
                temp_list = df_arc_task_sample_test[df_arc_task_sample_test['Name'] == file_name]['class'].tolist()[0].split(', ')
                for temp_tag in temp_list:
                    target_file_name.append(temp_tag)
        else:
            target_file_name = df_arc_task_sample[df_arc_task_sample['Name'] == file_name]['class'].tolist()
            if not target_file_name:
                continue
        padding_task = target_file_name

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

        test_task_input.append(input_arr)
        test_task_output.append(output_arr)

        test_task_input_size.append(input_size)
        test_task_output_size.append(output_size)

        # test_task_task_class.append(target_file_name)

    test_input.append(test_task_input)
    test_output.append(test_task_output)

    test_input_size.append(test_task_input_size)
    test_output_size.append(test_task_output_size)

    test_task_class.append(target_file_name)


    for i in range(len(valid_data)):
        valid_data[i]['input'] = (np.array(valid_data[i]['input'])).tolist()
        valid_input_y_len, valid_input_x_len = np.array(valid_data[i]['input']).shape
        valid_output_y_len, valid_output_x_len = np.array(valid_data[i]['output']).shape

        input_size = (valid_input_y_len, valid_input_x_len)
        output_size = (valid_output_y_len, valid_output_x_len)

        input_arr = [
            [0 if m > valid_input_x_len - 1 or n > valid_input_y_len - 1 else valid_data[i]['input'][n][m] + 1 for m
             in range(max_len)] for n in range(max_len)]
        output_arr = [
            [0 if m > valid_output_x_len - 1 or n > valid_output_y_len - 1 else valid_data[i]['output'][n][m] + 1
             for m in range(max_len)] for n in range(max_len)]

        # input_arr = valid_data[i]['input']
        # output_arr = valid_data[i]['output']
        #
        # test_input.append(input_arr)
        # # test_auto_dataset.append(input_arr)
        # test_output.append(output_arr)
        # # test_auto_dataset.append(output_arr)
        #
        # test_input_size.append(input_size)
        # # test_auto_size.append(input_size)
        # test_output_size.append(output_size)
        # # test_auto_size.append(output_size)
        # test_task_class.append(target_file_name)

# train_json_data = {
#     'input': train_input,
#     'output': train_output,
#     'input_size': train_input_size,
#     'output_size': train_output_size,
#     'task': train_task_class,
# }
#
# valid_json_data = {
#     'input': valid_input,
#     'output': valid_output,
#     'input_size': valid_input_size,
#     'output_size': valid_output_size,
#     'task': valid_task_class,
# }

test_json_data = {
    'input': test_input,
    'output': test_output,
    'input_size': test_input_size,
    'output_size': test_output_size,
    'task': test_task_class,
}

# train_dataframe = pd.DataFrame({
#     'input': train_input,
#     'output': train_output
# })
#
# valid_dataframe = pd.DataFrame({
#     'input': valid_input,
#     'output': valid_output
# })

# train_dataframe.to_csv('arc_train.csv', index=None)
# valid_dataframe.to_csv('arc_valid.csv', index=None)
# a = pd.read_csv('arc_train.csv')
# b = pd.read_csv('arc_valid.csv')





# with open(f'{dataset_path}/train_new_idea_task_sample2_.json', 'w') as f:
#     json.dump(train_json_data,f)
#
# with open(f'{dataset_path}/valid_new_idea_task_sample2_.json', 'w') as f:
#     json.dump(valid_json_data, f)

# with open(f'{dataset_path}/phase1_concept_train_common.json', 'w') as f:
#     json.dump(train_json_data, f)

with open(f'{dataset_path}/Concept_classification.json', 'w') as f:
    # json.dump(valid_json_data, f)
    json.dump(test_json_data, f)
