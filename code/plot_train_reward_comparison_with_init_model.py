# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import copy
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO


parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--my_seed', type = int, required = True)
parser.add_argument('--dataset', type = str, required = True)       # dataset = {   'sentiment-0', 'sentiment-1', 
                                                                    #               'toxicity-0', 'toxicity-1', 
                                                                    #               'politeness-0', 'politeness-1',
                                                                    #               'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                    #               'topic-0', 'act-1', 'act-2', 'act-3'}
parser.add_argument('--plot_model', type = str, required = False)   # plot_model : {'all', 'gpt2_small', 'opt', 'xglm', 'gpt2_large', 'gpt2_small_init_weight=uniform'}
parser.add_argument('--history', type = str, required = False)      # history : {'reward', 'acc', 'loss'}
args = parser.parse_args()
my_model = args.plot_model
my_dataset = args.dataset
my_history = args.history + '_history'

'''
데이터 주소 로드
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
FILE_DIR_LIST = []

# # 모든 model의 결과를 하나의 플롯에 뽑는다면
# if my_plot_model == 'all':
#     model_list = ['gpt2_small', 'opt', 'xglm', 'gpt2_large']

# # 각 model의 결과를 각 플롯에 뽑는다면
# else:
#     model_list = [my_plot_model]
#     model_size = my_model_size

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'

'''
파일 불러오기
'''
FILE_DIR_LIST = glob.glob(RESULT_DIR + '/{}/{}/*{}*{}*[!.pdf]'.format(my_dataset, my_model, 'stochastic', 'quantile'))

# Filter out the directory with 'quantile_0.85'
FILE_DIR_LIST = [dir for dir in FILE_DIR_LIST if 'quantile_0.85' not in dir]

'''
데이터 로드
'''
for idx, each_dir in enumerate(FILE_DIR_LIST):

    # my_model = each_dir.split('/')[8]                   # e.g., gpt2_small
    # my_dropout = each_dir.split('_')[-2]                # e.g., quantile
    my_dropout_rate = each_dir.split('_')[-1]           # e.g., 0.95
    
    train_history_pd = pd.read_csv(each_dir + '/' + my_history + '.csv', index_col=0)           # e.g., 
    # train_history_pd['model'] = my_model
    train_history_pd['dropout_rate'] = my_dropout_rate
    # print('model :', my_model)
    print('dropout_rate :', my_dropout_rate)
    # print('length : ', train_history_pd.shape)

    if idx == 0:
        train_history_pd_all = copy.deepcopy(train_history_pd)

    else:
        train_history_pd_all = pd.concat([train_history_pd_all, train_history_pd], axis=0)


dropout_rate_list = np.unique(train_history_pd_all['dropout_rate'])
epoch_len = train_history_pd.shape[0]

'''
전체 범례 설정
'''
# # Define labels and colors for the legend (already defined)
# legend_labels = ["q=0.0", "q=0.8", "q=0.85", "q=0.9", "q=0.95"]
# legend_colors = ['blue', 'orange', 'green', 'red', 'purple']

# # Create custom handles for the legend (already defined)
# custom_handles = [Line2D([0], [0], color=legend_colors[i], marker='o', linestyle='None',
#                          markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]

plt.figure(figsize = (3.5, 2.5))
for i, my_dropout_rate in enumerate(dropout_rate_list):

    first_filter = train_history_pd_all[train_history_pd_all['dropout_rate'] == my_dropout_rate]
    # target_history = first_filter['train_{}'.format(args.history)]
    target_history = first_filter['train_{}'.format(my_history.replace('_history', ''))]
    plt.plot(target_history, marker='.', markersize=10, label='{}_{}'.format('quantile', my_dropout_rate))

    if my_model == 'gpt2_small_init_weight=uniform' and my_history == 'reward_history':
        cmap = plt.get_cmap("tab10")
        max_val = np.max(target_history)
        plt.axhline(y=max_val, linestyle='solid', color=cmap(i))

# plt.legend()
plt.tight_layout()
plt.grid(True)
# plt.legend(handles=custom_handles, loc='lower center', 
#            ncol=len(custom_handles), bbox_to_anchor=(0.5, .1), 
#            fontsize=8, borderpad=.5)

plt.xticks(fontsize=8, ticks=np.arange(0, epoch_len, 1), labels=np.arange(1, epoch_len+1, 1))
plt.xlabel('epoch')

min_val = np.min(train_history_pd_all.iloc[:, 0])
max_val = np.max(train_history_pd_all.iloc[:, 0])

plt.yticks(fontsize=10, ticks=np.round(np.arange(min_val-0.05, max_val+0.05, 0.05), 1), labels=np.round(np.arange(min_val-0.05, max_val+0.05, 0.05), 1))
# plt.ylabel('mean {}'.format(my_history.replace('_history', '')))
# plt.ylabel('mean {}'.format('reward'))

# plt.legend(bbox_to_anchor = (1.0, 1.0), loc='upper left')
# plt.title('{}'.format(my_history.replace('_history', '')))
if my_history.replace('_history', '') == 'reward':
    plt.title('$R_{\phi}(\hat{\\tau}), \ \hat{\\tau} \sim \pi_{\\theta}$')
elif my_history.replace('_history', '') == 'acc':
    plt.title('$\\beta_{\\bar{\\theta}}(\hat{\\tau}), \ \hat{\\tau} \sim \pi_{\\theta}$')

plt.show()
plt.savefig(RESULT_DIR + '/{}'.format(my_dataset) + '/{}_comparison_20_epoch_plot_{}_{}_{}_{}.pdf'.format(my_model, my_dataset, 'stochastic', 'quantile', my_history), bbox_inches='tight')