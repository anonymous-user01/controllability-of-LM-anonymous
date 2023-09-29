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
parser.add_argument('--mode', type = str, required = True)

# mode == 'gen'/''test' 둘다 필요
parser.add_argument('--num_epi', type = int, required = True)
parser.add_argument('--min_reward_action', type = int, required = True)
parser.add_argument('--reward_order', type = int, required = True)

# mode == 'test' 일 때만 필요
parser.add_argument('--batch_size', type = int, required = False)
parser.add_argument('--lr', type = float, required = False)
parser.add_argument('--num_epoch', type = int, required = False)

args = parser.parse_args()

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])
# %%
# '''
# 1) subplot 방식
# - normal behavior policy (정규 행동정책 분포)
# - normal reward distribution (정규 보상분포)
# '''
# # reward_actions, reward_counts, bp_samples = get_samples(num_sample=sample_size, mu=9, sigma=1)
# sample_size = 100000
# epi_len = 10
# num_epi = sample_size//epi_len
# reward_dist = get_reward_dist(min_action=6, order=5)
# supports = np.arange(1, 11, 1)

# fig = plt.figure(figsize=(12, 8))
# for i, bp_mu in enumerate(range(2, 8, 1)):
#     bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=bp_mu, sigma=1)

#       .subplot(2, 3, i + 1)
#     plt.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
#     # plt.bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
#     plt.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward', color='red')
#     plt.xticks(ticks=supports, labels=supports)
#     plt.yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
#     plt.title('Case {} :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(i, bp_mu, 1))
#     if i == 0:
#         plt.legend(loc='upper left', fontsize=14)

# fig.tight_layout(pad=4.0)
# fig.supylabel('Frequency', fontsize=16)
# fig.supxlabel('Action', fontsize=16)
# plt.show()


# %%
# '''
# subplots 방식 (subplot"s"임 ! s가 붙은거 !)
# - normal behavior policy
# - normal reward distribution
# - alpha : action 인덱스 최솟값
# '''
# sample_size = 100000
# epi_len = 10
# num_epi = sample_size//epi_len
# reward_dist = get_reward_dist(min_action=6, order=5)
# supports = np.arange(1, 11, 1)

# nrows=2
# ncols=3
# # bp_mu_list = list(np.arange(3, 8, 2)) + [1, 9]
# bp_mu_list = np.arange(2, 8).reshape(2, 3)
# case_list = np.array(range(nrows*ncols)).reshape(2, 3)
# fig, ax1 = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)

# for i in range(nrows):
#     for j in range(ncols):
#         bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=bp_mu_list[j], sigma=1)
        
#         action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
#         ax1[i][j].set_xticks(ticks=supports, labels=supports)
#         ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
#         ax1[i][j].set_xlabel('action')
#         ax1[i][j].set_ylabel('likelihood')
#         ax1[i][j].set_title('Case {} :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], bp_mu_list[j], 1), fontsize=16)

#         # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
#         ax2 = ax1[i][j].twinx()
#         ax2.set_ylabel('reward', rotation=270, labelpad=15)
#         reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
#         if i == 0 and j == 0:
#             label1 = action_barplot.get_label()
#             label2 = reward_barplot.get_label()
#             ax1[i][j].legend([action_barplot, reward_barplot], [label1, label2], loc = 'upper left', fontsize=14)
# fig.tight_layout(pad=4.0)
# # fig.supxlabel('Action', fontsize=16)
# # fig.supylabel('Frequency', fontsize=16)
# # fig.supylabel('Reward', fontsize=16)

# %%
'''
2) subplots 방식
- double-mode normal behavior policy (쌍봉 정규 행동정책 분포)
- normal reward distribution (정규 보상분포)
- alpha : action 인덱스 최솟값
'''
num_epi = args.num_epi
epi_len = 10
sample_size = num_epi * epi_len
reward_dist = get_reward_dist(min_action=args.min_reward_action, order=args.reward_order)

supports = np.arange(1, 11, 1)
# nrows=3
# ncols=3
nrows=3
ncols=2
case_list = np.array(range(nrows*ncols)).reshape(ncols, nrows).T
num_cases = len(np.concatenate(case_list))
bp_mu_list = np.array([3, 5, 7])
bp_sigma_list = np.array([0.5, 1.0, 3.0])
all_samples = np.zeros((1, 0)).astype('int32')


'''
gen 모드
'''
if args.mode == 'gen':

    '''
    gen 및 test 모드일 때 공통적으로 필요한 generating & plotting 동작 수행
    '''
    fig, ax1 = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)

    # for i in range(nrows):
    for j in range(ncols):

        k = 0

        # for j in range(ncols):
        for i in range(nrows):

            # if i == 0:
            if j == 0:
                '''
                행동정책이 Normal 분포를 따르는 case
                '''

                # behavior-policy 분포를 따라 샘플링
                bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=bp_mu_list[i], sigma=0.5, dist='truncnorm')

                # behavior-policy 분포 플로팅
                action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
                ax1[i][j].set_xticks(ticks=supports, labels=supports)
                # ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
                ax1[i][j].set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1))
                ax1[i][j].set_xlabel('action')
                ax1[i][j].set_ylabel('likelihood')
                ax1[i][j].set_title('Case {} (different $\mu$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], bp_mu_list[i], 0.5), fontsize=16)

                # reward 분포 플로팅
                # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
                ax2 = ax1[i][j].twinx()
                ax2.set_ylabel('reward', rotation=270, labelpad=15)
                reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
    
                # 첫번째 row, 첫번째 col의 plot에만 범례 달기
                # if i == 0 and j == 0:
                if i == 0:
                    label1 = action_barplot.get_label()
                    label2 = reward_barplot.get_label()
                    ax1[i][j].legend([action_barplot, reward_barplot], [label1, label2], loc = 'upper left', fontsize=14)

                # elif i == 0 and j == 1:
                elif i == 1:
                    ax1[i][j].text(0.01, 0.99, "N={}".format(num_epi), ha='left', va='top', transform=ax1[i][j].transAxes, fontsize=16)
                    ax1[i][j].text(0.01, 0.89, "L={}".format(epi_len), ha='left', va='top', transform=ax1[i][j].transAxes, fontsize=16)

                # all_samples에 행동정책의 행동샘플 추가
                all_samples = np.append(all_samples, bp_samples)

            # elif i == 1:
            elif j == 1:
                '''
                행동정책이 Normal 분포를 따르는 case
                '''

                # behavior-policy 분포를 따라 샘플링
                bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=5, sigma=bp_sigma_list[i], dist='truncnorm')

                # behavior-policy 분포 플로팅
                action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
                ax1[i][j].set_xticks(ticks=supports, labels=supports)
                # ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
                ax1[i][j].set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1))
                ax1[i][j].set_xlabel('action')
                ax1[i][j].set_ylabel('likelihood')
                ax1[i][j].set_title('Case {} (different $\sigma$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], 5, bp_sigma_list[i]), fontsize=16)

                # reward 분포 플로팅
                # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
                ax2 = ax1[i][j].twinx()
                ax2.set_ylabel('reward', rotation=270, labelpad=15)
                reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
    
                # all_samples에 행동정책의 행동샘플 추가
                all_samples = np.append(all_samples, bp_samples)

            # elif i == 2:

            #     if j == 0:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - left-skewed Poisson 분포를 따르는 case
            #         '''
            #         lambda_ = 1.5

            #         # behavior-policy 분포를 따라 샘플링 (left-skewed)
            #         bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=lambda_, sigma=None, dist='truncpoisson')

            #         # behavior-policy 분포 플로팅
            #         action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         ax1[i][j].set_xlabel('action')
            #         ax1[i][j].set_ylabel('likelihood')
            #         ax1[i][j].set_title('Case {} (different $\gamma$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=16)

            #         # reward 분포 플로팅
            #         # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
            #         ax2 = ax1[i][j].twinx()
            #         ax2.set_ylabel('reward', rotation=270, labelpad=15)
            #         reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')

            #         # all_samples에 행동정책의 행동샘플 추가
            #         all_samples = np.append(all_samples, bp_samples)

            #     elif j == 1:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - middle-skewed Poisson 분포를 따르는 case
            #         '''
            #         lambda_ = 5.5

            #         # behavior-policy 분포를 따라 샘플링 (right-skewed)
            #         bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=lambda_, sigma=None, dist='truncpoisson')

            #         # behavior-policy 분포 플로팅
            #         action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         ax1[i][j].set_xlabel('action')
            #         ax1[i][j].set_ylabel('likelihood')
            #         ax1[i][j].set_title('Case {} (different $\gamma$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=16)

            #         # reward 분포 플로팅
            #         # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
            #         ax2 = ax1[i][j].twinx()
            #         ax2.set_ylabel('reward', rotation=270, labelpad=15)
            #         reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')

            #         # all_samples에 행동정책의 행동샘플 추가
            #         all_samples = np.append(all_samples, bp_samples)

            #     elif j == 2:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - right-skewed Poisson 분포를 따르는 case
            #         '''
            #         lambda_ = 15
                    
            #         # behavior-policy 분포를 따라 샘플링 (right-skewed)
            #         bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=lambda_, sigma=None, dist='truncpoisson')

            #         # behavior-policy 분포 플로팅
            #         action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         ax1[i][j].set_xlabel('action')
            #         ax1[i][j].set_ylabel('likelihood')
            #         ax1[i][j].set_title('Case {} (different $\gamma$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=16)

            #         # reward 분포 플로팅
            #         # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
            #         ax2 = ax1[i][j].twinx()
            #         ax2.set_ylabel('reward', rotation=270, labelpad=15)
            #         reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')

            #         # all_samples에 행동정책의 행동샘플 추가
            #         all_samples = np.append(all_samples, bp_samples)

                # elif j == 2:
                #     '''
                #     행동정책이 double-mode Normal 분포를 따르는 case
                #     - 쌍봉 행동정책
                #     - Forward KL 방식의 약점을 보여줌
                #     '''

                #     first_mode_mu = 2
                #     second_mode_mu = first_mode_mu+5

                #     # behavior-policy 분포를 따라 샘플링
                #     bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, mu=first_mode_mu, sigma=1, dist='trunc_doublenorm')

                #     # behavior-policy 분포 플로팅
                #     action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
                #     ax1[i][j].set_xticks(ticks=supports, labels=supports)
                #     ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
                #     ax1[i][j].set_xlabel('action')
                #     ax1[i][j].set_ylabel('likelihood')
                #     ax1[i][j].set_title('Case {} :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$ \n $ \ \ +truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], first_mode_mu, 1, second_mode_mu, 1), fontsize=16)

                #     # reward 분포 플로팅
                #     # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
                #     ax2 = ax1[i][j].twinx()
                #     ax2.set_ylabel('reward', rotation=270, labelpad=15)
                #     reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
        
                #     # 첫번째 row, 첫번째 col의 plot에만 범례 달기
                #     if i == 0 and j == 0:
                #         label1 = action_barplot.get_label()
                #         label2 = reward_barplot.get_label()
                #         ax1[i][j].legend([action_barplot, reward_barplot], [label1, label2], loc = 'upper left', fontsize=14)

                #     # all_samples에 행동정책의 행동샘플 추가
                #     all_samples = np.append(all_samples, bp_samples)

                # '''
                # 행동정책이 double-mode Normal 분포를 따르는 case
                # - 쌍봉 행동정책
                # - Forward KL 방식의 약점을 보여줌
                # '''

                # # first_mode_mu = 2
                # mu1 = 2 + j
                # mu2 = 5 + j + k
                # sigma1 = sigma2 = 1
                # k += 1
                # sample_ratio1 = round(1 - (j + 2) * 0.1, 2)
                # sample_ratio2 = round(1 - sample_ratio1, 2)

                # # behavior-policy 분포를 따라 샘플링
                # bp_actions, bp_counts, bp_samples = get_samples(num_sample=sample_size, sigma=1, dist='trunc_doublenorm', sample_ratio=sample_ratio1, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)

                # # behavior-policy 분포 플로팅
                # action_barplot = ax1[i][j].bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
                # ax1[i][j].set_xticks(ticks=supports, labels=supports)
                # ax1[i][j].set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
                # ax1[i][j].set_xlabel('action')
                # ax1[i][j].set_ylabel('likelihood')
                # ax1[i][j].set_title('Case {} :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$ \n $ \ \ +truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], mu1, sigma1, mu2, sigma2), fontsize=16)

                # # 첫번째 row, 첫번째 col의 plot에만 범례 달기
                # ax1[i][j].text(0.01, 0.99, "$\\rho_1$={}".format(sample_ratio1), fontsize=16, ha='left', va='top', transform=ax1[i][j].transAxes)
                # ax1[i][j].text(0.01, 0.89, "$\\rho_2$={}".format(sample_ratio2), fontsize=16, ha='left', va='top', transform=ax1[i][j].transAxes)


                # # reward 분포 플로팅
                # # ax[i][j].bar(reward_actions, reward_counts, align='center', width = .7, alpha=0.5, label='reward', color='red')
                # ax2 = ax1[i][j].twinx()
                # ax2.set_ylabel('reward', rotation=270, labelpad=15)
                # reward_barplot = ax2.bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')

                # # all_samples에 행동정책의 행동샘플 추가
                # all_samples = np.append(all_samples, bp_samples)

    fig.tight_layout(pad=1.)
    fig.savefig(parent_dir + '/figure1.pdf')


    # 최종 synthetic data 및 train_reward_dist 만들기
    case_labels = np.repeat(np.arange(num_cases), sample_size)
    all_data = np.concatenate([all_samples[:, np.newaxis], case_labels[:, np.newaxis]], axis=-1)
    all_data_pd = pd.DataFrame(all_data, columns=["bp_sampled_actions", "case"])
    all_data_pd.to_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order))
    train_reward_dist = pd.DataFrame(reward_dist, columns=["reward_per_action"])
    train_reward_dist.index = train_reward_dist.index+1
    train_reward_dist.to_csv(parent_dir + '/prep_data/position-game/train_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order))

# %%

'''
test 모드
'''
if args.mode == 'test':

    batch_size = args.batch_size
    lr = args.lr
    num_epoch = args.num_epoch

    '''
    get 데이터 불러오기
    '''
    synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)    

    '''
    플롯팅
    '''
    fig, ax1 = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)

    # for i in range(nrows):
    for j in range(ncols):

        k = 0

        # for j in range(ncols):
        for i in range(nrows):
                     
            '''
            gen 데이터 case 별로 분리하기
            '''    
            target_case = case_list[i][j]
            syn_data_case = synthetic_data[synthetic_data['case'] == target_case]
            bp_actions, bp_counts = np.unique(syn_data_case['bp_sampled_actions'], return_counts = True)


            '''
            test 데이터 불러오기 및 case 별로 분리하기
            '''
            # file_dir = str(batch_size) + '_' + str(lr) + '_' + str(num_epoch) + '_' + str(target_case)
            file_dir = '{}_{}_{}_{}_mra={}_ro={}'.format(batch_size, lr, num_epoch, target_case, args.min_reward_action, args.reward_order)
            print('file_dir :', file_dir)
            all_test_action_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_action_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
            tp_actions = all_test_action_dist_pd.columns
            tp_actions = np.array(tp_actions).astype('int32')
            tp_counts = all_test_action_dist_pd.iloc[0]
            tp_counts = np.array(tp_counts).astype('int32')

            '''
            test의 보상 결과 데이터 불러오기
            '''
            print('file_dir :', file_dir)
            all_test_reward_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
            mean_test_reward = tf.reduce_mean(all_test_reward_dist_pd)

            '''
            모멘트 분석
            '''
            # if i == 0:
            if j == 0:

                '''
                1차 모멘트 분석
                '''

                # reward 분포 플로팅
                reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
                ax1[i][j].set_xticks(ticks=supports, labels=supports, fontsize=16)
                ax1[i][j].set_yticks(ticks=np.round(np.arange(0, 1.2, 0.2), 1), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=16)
                ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15, fontsize=16)

                # behavior-policy 분포 플로팅
                ax2 = ax1[i][j].twinx()
                bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
                ax2.set_xlabel('action', fontsize=16)
                ax2.set_ylabel('likelihood', fontsize=16)

                # target policy 분포 플로팅
                tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
                # ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)], fontsize=16)
                ax2.set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=16)
                for idx, tp_bar in enumerate(tp_action_barplot):
                    tp_action_barplot[idx].set_hatch("/" * 5)

                # 첫번째 row, 첫번째 col의 plot에만 범례 달기
                # if j == 0:
                if i == 0:
                    label1 = bp_action_barplot.get_label()
                    label2 = reward_barplot.get_label()
                    # ax2.set_title('Case {} (ill-posed $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.legend([bp_action_barplot, reward_barplot], [label1, label2], loc = 'upper left', fontsize=16)

                    # 평균 보상
                    ax2.text(0.035, 0.5, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

                # elif j == 1:
                elif i == 1:
                    # ax2.set_title('Case {} (partially well-posed $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.legend([tp_action_barplot], [tp_action_barplot.get_label()], loc = 'upper left', fontsize=16)

                    # 평균 보상
                    ax2.text(0.035, 0.65, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

                # elif j == 2:
                elif i == 2:
                    # ax2.set_title('Case {} (well-posed $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, bp_mu_list[i], 0.5), fontsize=18)
                    ax2.text(0.035, 0.95, "N={}".format(num_epi), ha='left', va='top', transform=ax2.transAxes, fontsize=16)
                    ax2.text(0.035, 0.80, "L={}".format(epi_len), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

                    # 평균 보상
                    ax2.text(0.035, 0.65, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

            # elif i == 1:
            elif j == 1:
                '''
                2차 모멘트 분석
                '''

                # reward 분포 플로팅
                reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
                ax1[i][j].set_xticks(ticks=supports, labels=supports, fontsize=16)
                ax1[i][j].set_yticks(ticks=np.round(np.arange(0, 1.2, 0.2), 1), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=16)
                ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15, fontsize=16)

                # behavior-policy 분포 플로팅
                ax2 = ax1[i][j].twinx()
                bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue') 
                ax2.set_xlabel('action', fontsize=16)
                ax2.set_ylabel('likelihood', fontsize=16)    

                # if j == 0:
                if i == 0:
                    # ax2.set_title('Case {} (narrow-ranged $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)
                # elif j == 1:
                elif i == 1:
                    # ax2.set_title('Case {} (mid-ranged $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)
                # elif j == 2:
                elif i == 2:
                    # ax2.set_title('Case {} (wide-ranged $\\beta$) :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)
                    ax2.set_title('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, 5, bp_sigma_list[i]), fontsize=18)

                # target policy 분포 플로팅
                tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
                # ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)], fontsize=16)
                ax2.set_yticks(ticks=np.arange(0, sample_size + 2 * num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=16)
                for idx, tp_bar in enumerate(tp_action_barplot):
                    tp_action_barplot[idx].set_hatch("/" * 5)

                # 평균 보상
                ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

            # elif i == 2:
            #     '''
            #     더블 모드 분석
            #     '''
                
            #     mu1 = 2 + j
            #     mu2 = 5 + j + k
            #     sigma1 = sigma2 = 1
            #     k += 1
            #     sample_ratio1 = round(1 - (j + 2) * 0.1, 2)
            #     sample_ratio2 = round(1 - sample_ratio1, 2)

            #     # reward 분포 플로팅
            #     reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
            #     ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #     ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15)

            #     # behavior-policy 분포 플로팅
            #     ax2 = ax1[i][j].twinx()
            #     bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')                
            #     ax2.set_xlabel('action')
            #     ax2.set_ylabel('likelihood')
            #     ax2.set_title('Case {} :\n $p_\\beta \sim truncNormal(\mu={}, \sigma={})$ \n $ \ \ +truncNormal(\mu={}, \sigma={})$'.format(case_list[i][j], mu1, sigma1, mu2, sigma2), fontsize=18)

            #     # 분포 비율 text 달기
            #     ax2.text(0.035, 0.95, "$\\rho_1$={}".format(sample_ratio1), fontsize=18, ha='left', va='top', transform=ax2.transAxe4)
            #     ax2.text(0.035, 0.80, "$\\rho_2$={}".format(sample_ratio2), fontsize=18, ha='left', va='top', transform=ax2.transAxe4)

            #     # target policy 분포 플로팅
            #     tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
            #     ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #     for idx, tp_bar in enumerate(tp_action_barplot):
            #         tp_action_barplot[idx].set_hatch("/" * 5)

            #     # 평균 보상
            #     ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

            # elif i == 2:
            #     '''
            #     3차 모멘트 분석
            #     '''

            #     if j == 0:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - left-skewed Poisson 분포를 따르는 case
            #         '''
            #         # reward 분포 플로팅
            #         reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15)

            #         # behavior-policy 분포 플로팅
            #         lambda_ = 1.5
            #         ax2 = ax1[i][j].twinx()
            #         bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            #         ax2.set_xlabel('action')
            #         ax2.set_ylabel('likelihood')
            #         ax2.set_title('Case {} (left-skewed $\\beta$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=18)


            #         # target policy 분포 플로팅
            #         tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
            #         ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         for idx, tp_bar in enumerate(tp_action_barplot):
            #             tp_action_barplot[idx].set_hatch("/" * 5)

            #         # 평균 보상
            #         ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)


            #     elif j == 1:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - middle-skewed Poisson 분포를 따르는 case
            #         '''
            #         # reward 분포 플로팅
            #         reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15)

            #         # behavior-policy 분포 플로팅
            #         lambda_ = 5.5
            #         ax2 = ax1[i][j].twinx()
            #         bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')                    
            #         ax2.set_xlabel('action')
            #         ax2.set_ylabel('likelihood')
            #         ax2.set_title('Case {} (balanced $\\beta$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=18)

            #         # target policy 분포 플로팅
            #         tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
            #         ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         for idx, tp_bar in enumerate(tp_action_barplot):
            #             tp_action_barplot[idx].set_hatch("/" * 5)

            #         # 평균 보상
            #         ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)


            #     elif j == 2:
            #         '''
            #         행동정책이 Poisson 분포를 따르는 case
            #         - right-skewed Poisson 분포를 따르는 case
            #         '''
                    
            #         # reward 분포 플로팅
            #         reward_barplot = ax1[i][j].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
            #         ax1[i][j].set_xticks(ticks=supports, labels=supports)
            #         ax1[i][j].set_ylabel('reward', rotation=270, labelpad=15)

            #         # behavior-policy 분포 플로팅
            #         lambda_ = 15
            #         ax2 = ax1[i][j].twinx()
            #         bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
            #         ax2.set_xlabel('action')
            #         ax2.set_ylabel('likelihood')
            #         ax2.set_title('Case {} (right-skewed $\\beta$) :\n $p_\\beta \sim truncPois(\lambda={})$'.format(case_list[i][j], lambda_), fontsize=18)

            #         # target policy 분포 플로팅
            #         tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
            #         ax2.set_yticks(ticks=np.arange(0, sample_size, num_epi), labels=[str(i/sample_size) for i in np.arange(0, sample_size, num_epi)])
            #         for idx, tp_bar in enumerate(tp_action_barplot):
            #             tp_action_barplot[idx].set_hatch("/" * 5)

            #         # 평균 보상
            #         ax2.text(0.035, 0.95, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=16)

    fig.tight_layout(pad=1.)
    fig.savefig(parent_dir + '/figure' + file_dir + '.pdf')
