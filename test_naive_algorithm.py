import json

train_path = 'data/phase1_concept_train_fusion_layer_v1.json'
valid_path = 'data/phase1_concept_valid_fusion_layer_v1.json'
# train_path = 'data/train_new_idea_task_concept_v2.json'
# valid_path = 'data/valid_new_idea_task_concept_v2.json'

with open(train_path, 'r') as f:
    train_data = json.load(f)

with open(valid_path, 'r') as f:
    valid_data = json.load(f)

train_max_string = None
train_max_count = 0
train_set = set()
if not type(train_data['task'][0]) == type(['a', 'b']):
    train_data['task'] = [i.split(',') for i in train_data['task']]
train_list = [' '.join(i) for i in train_data['task']]
total_train_count = 0
for i in train_data['task']:
    train_set.update([' '.join(i)])

for i in list(train_set):
    count = train_list.count(i)
    total_train_count += count
    print(f'{i}: {count}')
    if train_max_count < count:
        train_max_string = i
        train_max_count = count

if total_train_count == len(train_list):
    print(len(train_set))
    print('\n--------train pass--------')
print('\n*****************************************\n')

valid_max_string = None
valid_max_count = 0
valid_set = set()
try:
    if not type(valid_data['task'][0]) == type(['a']):
        valid_data['task'] = [i.split(',') for i in valid_data['task']]
except:
    print(1)
valid_list = [' '.join(i) for i in valid_data['task']]
total_valid_count = 0

for i in valid_data['task']:
    valid_set.update([' '.join(i)])

for i in list(valid_set):
    count = valid_list.count(i)
    total_valid_count += count
    print(f'{i}: {count}')
    if valid_max_count < count:
        valid_max_string = i
        valid_max_count = count

if total_valid_count == len(valid_list):
    print(len(valid_set))
    print('\n--------valid pass--------')

print('\n*****************************************\n')

print(f'High train frequency pattern: ({train_max_string}, {train_max_count})')
print(f'High valid frequency pattern: ({valid_max_string}, {valid_max_count})')
print(f'Apply high train frequency pattern about valid: ({train_max_string}, {valid_list.count(train_max_string)})  -> {valid_list.count(train_max_string) / len(valid_list)}')