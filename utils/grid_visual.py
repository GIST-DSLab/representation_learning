import matplotlib
import numpy as np
from matplotlib import colors    ### 까만 노드 인력척력 djqt는 버전 + grid_to_graph_woblack 함수 추가 + 1.414 주석처리 + np array 로 변경
import matplotlib.pyplot as plt
from io import BytesIO
import base64

cmap = colors.ListedColormap(
        [
            '#000000', # 0 검은색
            '#0074D9', # 1 파란색
            '#FF4136', # 2 빨간색
            '#2ECC40', # 3 초록색
            '#FFDC00', # 4 노란색
            '#AAAAAA', # 5 회색
            '#F012BE', # 6 핑크색
            '#FF851B', # 7 주황색
            '#7FDBFF', # 8 하늘색
            '#870C25', # 9 적갈색
            '#505050', # 10 검은색_select
            '#30A4F9', # 11 파란색_select
            #'#FF4136',
            '#FF7166', # 12 빨간색_select
            '#5EFC70', # 13 초록색_select
            '#FFFC30', # 14 노란색_select
            '#DADADA', # 15 회색_select
            '#F042EE', # 16 핑크색_select
            '#FFB54B', # 17 주황색_select
            '#AFFBFF', # 18 하늘색_select
            '#B73C55'  # 19 적갈색_select
        ])
    #norm = colors.Normalize(vmin=0, vmax=9)
norm = colors.Normalize(vmin=0, vmax=19)

def string_to_array(grid):
    # if grid is already in integer form, just return it
    if isinstance(grid[0][0], int): return grid

    mapping = {0:'.',1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'g',8:'h',9:'i',10:'j'}
    revmap = {v:k for k,v in mapping.items()}
    newgrid = [[revmap[grid[i][j]] for j in range(len(grid[0]))] for i in range(len(grid))]
    return newgrid

def plot_2d_grid(data):
    cvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    colors = ["black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown",
              "black"]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    fig, axs = plt.subplots(len(data['test']), 3, figsize=(5, len(data['test']) * 3 * 0.7))
    axs = axs.reshape(-1, 3)  # Reshape axs to have 2 dimensions

    # show grid
    for i, example in enumerate(data['test']):
        axs[i, 0].set_title(f'Test Input {i + 1}')
        # display gridlines
        rows, cols = np.array(string_to_array(example['input'])).shape
        axs[i, 0].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[i, 0].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[i, 0].grid(True, which='minor', color='black', linewidth=0.5)
        axs[i, 0].set_xticks([]);
        axs[i, 0].set_yticks([])
        axs[i, 0].imshow(np.array(string_to_array(example['input'])), cmap=cmap, vmin=0, vmax=9)

        axs[i, 1].set_title(f'Test Output {i + 1}')
        # display gridlines
        rows, cols = np.array(string_to_array(example['output'])).shape
        axs[i, 1].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[i, 1].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[i, 1].grid(True, which='minor', color='black', linewidth=0.5)
        axs[i, 1].set_xticks([]);
        axs[i, 1].set_yticks([])
        axs[i, 1].imshow(np.array(string_to_array(example['output'])), cmap=cmap, vmin=0, vmax=9)
        # plot gpt output if present
        if 'code_output' in example:
            axs[i, 2].set_title(f'GPT Output {i + 1}')
            # display gridlines
            rows, cols = np.array(string_to_array(example['code_output'])).shape
            axs[i, 2].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
            axs[i, 2].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
            axs[i, 2].grid(True, which='minor', color='black', linewidth=0.5)
            axs[i, 2].set_xticks([]);
            axs[i, 2].set_yticks([])
            axs[i, 2].imshow(np.array(string_to_array(example['code_output'])), cmap=cmap, vmin=0, vmax=9)
        else:
            axs[i, 2].axis('off')
    plt.tight_layout()

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    plt.show()

    fig, axs = plt.subplots(len(data['train']), 3, figsize=(5, len(data['train']) * 3 * 0.7))
    axs = axs.reshape(-1, 3)  # Reshape axs to have 2 dimensions
    for i, example in enumerate(data['train']):
        axs[i, 0].set_title(f'Training Input {i + 1}')
        # display gridlines
        rows, cols = np.array(string_to_array(example['input'])).shape
        axs[i, 0].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[i, 0].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[i, 0].grid(True, which='minor', color='black', linewidth=0.5)
        axs[i, 0].set_xticks([]);
        axs[i, 0].set_yticks([])
        axs[i, 0].imshow(np.array(string_to_array(example['input'])), cmap=cmap, vmin=0, vmax=9)

        axs[i, 1].set_title(f'Training Output {i + 1}')
        # display gridlines
        rows, cols = np.array(string_to_array(example['output'])).shape
        axs[i, 1].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
        axs[i, 1].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
        axs[i, 1].grid(True, which='minor', color='black', linewidth=0.5)
        axs[i, 1].set_xticks([]);
        axs[i, 1].set_yticks([])
        axs[i, 1].imshow(np.array(string_to_array(example['output'])), cmap=cmap, vmin=0, vmax=9)
        if 'code_output' in example:
            axs[i, 2].set_title(f'GPT Output {i + 1}')
            # display gridlines
            rows, cols = np.array(string_to_array(example['code_output'])).shape
            axs[i, 2].set_xticks(np.arange(cols + 1) - 0.5, minor=True)
            axs[i, 2].set_yticks(np.arange(rows + 1) - 0.5, minor=True)
            axs[i, 2].grid(True, which='minor', color='black', linewidth=0.5)
            axs[i, 2].set_xticks([]);
            axs[i, 2].set_yticks([])
            axs[i, 2].imshow(np.array(string_to_array(example['code_output'])), cmap=cmap, vmin=0, vmax=9)
        else:
            axs[i, 2].axis('off')
    plt.tight_layout()

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    plt.show()

    # returns back in html format
    return html