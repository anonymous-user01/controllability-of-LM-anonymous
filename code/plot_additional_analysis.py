# %%%
import os
from pathlib import Path
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from utils import get_samples, get_reward_dist
import copy
import random

'''
시드 설정
'''
os.environ["PYTHONHASHSEED"] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(47)
np.random.seed(47)
random.seed(47)

'''
파라미터 설정
'''
parser = argparse.ArgumentParser(description='receive the parameters')
# parser.add_argument('--mode', type = str, required = True)

# mode == 'gen'/''test' 둘다 필요
parser.add_argument('--num_epi', type = int, required = True)
parser.add_argument('--min_reward_action', type = int, required = True)
# parser.add_argument('--reward_order', type = int, required = True)

# mode == 'test' 일 때만 필요
parser.add_argument('--batch_size', type = int, required = False)
parser.add_argument('--lr', type = float, required = False)
parser.add_argument('--num_epoch', type = int, required = False)

args = parser.parse_args()

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
2) subplots 방식
- double-mode normal behavior policy (쌍봉 정규 행동정책 분포)
- normal reward distribution (정규 보상분포)
- alpha : action 인덱스 최솟값
'''
num_epi = args.num_epi
epi_len = 10
sample_size = num_epi * epi_len

nrows=1
ncols=4
case_list = np.array(range(nrows*ncols)).reshape(ncols, nrows)
num_cases = len(np.concatenate(case_list))
supports = np.arange(1, 11, 1)
batch_size = args.batch_size
lr = args.lr
num_epoch = args.num_epoch
target_case = 5

'''
플롯팅
'''
fig, ax1 = plt.subplots(figsize=(12, 3), nrows=nrows, ncols=ncols, squeeze=False)
for subplot_idx, reward_order in enumerate([1, 2, 4, 16]):
    reward_dist = get_reward_dist(min_action=args.min_reward_action, order=reward_order)

    '''
    gen 데이터 불러오기
    '''
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, reward_order), index_col=0)

    '''
    gen 데이터 case 별로 분리하기
    '''    
    syn_data_case = synthetic_data[synthetic_data['case'] == target_case]
    bp_actions, bp_counts = np.unique(syn_data_case['bp_sampled_actions'], return_counts = True)

    '''
    test 데이터 불러오기 및 case 별로 분리하기
    '''
    file_dir = '{}_{}_{}_{}_mra={}_ro={}'.format(batch_size, lr, num_epoch, target_case, args.min_reward_action, reward_order)
    print('file_dir :', file_dir)
    all_test_action_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_action_dist_mra={}_ro={}.csv'.format(args.min_reward_action, reward_order), index_col=0)
    tp_actions = all_test_action_dist_pd.columns
    tp_actions = np.array(tp_actions).astype('int32')
    tp_counts = all_test_action_dist_pd.iloc[0]
    tp_counts = np.array(tp_counts).astype('int32')

    '''
    test의 보상 결과 데이터 불러오기
    '''
    print('file_dir :', file_dir)
    all_test_reward_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, reward_order), index_col=0)
    mean_test_reward = tf.reduce_mean(all_test_reward_dist_pd)

    # reward 분포 플로팅
    reward_barplot = ax1[0][subplot_idx].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
    ax1[0][subplot_idx].set_xticks(ticks=supports, labels=supports, fontsize=14)
    # ax1[0][subplot_idx].set_yticks(ticks=np.round(reward_dist, 1), labels=np.round(reward_dist, 1))
    ax1[0][subplot_idx].set_yticks(ticks=np.round(np.arange(0, 1.2, 0.2), 1), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=14)
    ax1[0][subplot_idx].set_ylabel('reward', rotation=270, labelpad=15, fontsize=14)

    # behavior-policy 분포 플로팅
    ax2 = ax1[0][subplot_idx].twinx()
    bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
    ax2.set_xlabel('action', fontsize=14)
    ax2.set_ylabel('likelihood', fontsize=14)

    # target policy 분포 플로팅
    tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
    for idx, tp_bar in enumerate(tp_action_barplot):
        tp_action_barplot[idx].set_hatch("/" * 5)
    # ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
    ax2.set_yticks(ticks=np.arange(0, sample_size + num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=14)

    # 평균 보상
    ax2.text(0.035, 0.975, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=14)

    # 서브플롯 타이틀
    title_list = ['$2^0$', '$2^1$', '$2^2$', '$2^4$']
    title_list2 = ['linear', 'quadratic', 'quartic', '16-th']
    ax2.set_title('Case {}:\n {}-th degree ({})'.format(int(subplot_idx), title_list[subplot_idx], title_list2[subplot_idx]), fontsize=14)

# # 수퍼 타이틀
# fig.suptitle('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, 5, 0.5), fontsize=16)

# 레이아웃 맞춤
fig.tight_layout(pad=1.)

# 저장하기
fig.savefig(parent_dir + '/4col_figure' + file_dir + '.pdf')
