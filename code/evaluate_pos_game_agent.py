# %%
import os
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import pandas as pd
import pickle, json
import copy
import time
from model_agent import TargetPolicy
from utils import *

# if __name__ == '__main'__:

print('\n')
print('\n')

'''
파라미터 설정
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--batch_size', type = int, required = True)
parser.add_argument('--lr', type = float, required = True)
parser.add_argument('--num_epoch', type = int, required = True)
parser.add_argument('--case', type = int, required = True)
parser.add_argument('--min_reward_action', type = int, required = True)
parser.add_argument('--reward_order', type = int, required = True)
args = parser.parse_args()

parent_dir = str(Path(os.getcwd()).parents[0])
synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
kwargs = {
    'model_name' : 'TargetPolicy',
    'batch_size' : args.batch_size,
    'action_size' : 10,
    'epi_len' : 10,
    'lr' : args.lr,
    'num_epoch' : args.num_epoch,
    'task' : 'position-game',
    'num_cases' : len(np.unique(synthetic_data['case'])),
    'case' : args.case
    # 'case' : 4
}

model_name = str(kwargs['model_name'])
batch_size_path = str(kwargs['batch_size'])
epoch_path = '_' + str(kwargs['num_epoch'])
lr_path = '_' + str(kwargs['lr'])
case = kwargs['case']
case_path = '_' + str(case)

'''
데이터 로드
'''
synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
reward_dist = pd.read_csv(parent_dir + '/prep_data/position-game/train_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
reward_dist = np.concatenate(np.array(reward_dist))
train_inputs, _, _ = get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs)
train_inputs_of_case = train_inputs[case]
sample_size = train_inputs_of_case.shape[0]
# sample_size = 500
epi_len = train_inputs_of_case.shape[1]
action_size = train_inputs_of_case.shape[2]

'''
테스트 데이터 생성
- action_feq_per_sample : 각 샘플 별 action의 샘플링 빈도수 (sample_size, action_size)
- most_freq_action_per_sample : 각 샘플 별 가장 샘플링 빈도가 높은 action (sample_size, )
- list_of_actions : 최빈도 action의 리스트
==> 그때그때 리스트를 구성하는 action의 조합 (즉 리스트의 형상)이 다름
==> train_inputs 생성 시 설정한 behavior policy의 분포에 따라 list_of_action에 담기는 action의 풀이 달라짐

- bp_mu_action : 각 case 별 behavior policy의 평균 행동 (mu_action)
- test_initial_state : step 0에 해당하는 상태벡터
==> test 단계 (evaluate 단계)에서는 시작점인 step 0의 상태가 [0, 0, .., 0] 벡터가 아닌 bp_mu_action에 의해 정의됨

- test_inputs : 테스트 데이터
'''
action_freq_per_sample = tf.reduce_sum(train_inputs_of_case, axis = 1)                    # action_freq : (sample_size, action_size)
most_freq_action_per_sample = tf.argmax(action_freq_per_sample, axis = -1)                # most_freq_action_per_sample : (sample_size, )
list_of_actions = tf.unique_with_counts(tf.sort(most_freq_action_per_sample, direction = 'ASCENDING'))[0].numpy()
counts_of_actions = tf.unique_with_counts(tf.sort(most_freq_action_per_sample, direction = 'ASCENDING'))[2].numpy()
bp_mu_action = list_of_actions[tf.argmax(counts_of_actions)]

test_initial_state = tf.one_hot(tf.cast(tf.ones(shape = (sample_size, )) * bp_mu_action, dtype = tf.int32), depth = action_size)        # test_initial_state : (sample, action_size)
test_inputs = tf.expand_dims(test_initial_state, axis = 1)                                                                                  # test_inputs : (sample, 1, action_size)

'''
저장 경로 생성
'''
SAVE_PATH_RESULT = set_SavePath(kwargs, save_file = 'results')  # 결과 저장 경로

# # 텐서플로 데이터셋 객체로 변환
# with tf.device("/cpu:0"):

#     # 학습 데이터
#     test_dataset = tf.data.Dataset.from_tensor_slices(test_inputs)
#     # test_batchset = test_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
#     test_batchset = test_dataset.batch(batch_size = test_inputs.shape[0], drop_remainder = True, num_parallel_calls = 8)
#     test_batchset = test_batchset.prefetch(1)


'''
pos_game_agent 모델 로드
'''
# 훈련 파라미터 로드
SAVE_PATH_PARAM = set_SavePath(kwargs, save_file = 'params')    # 파라미터 저장 경로
param_dir = SAVE_PATH_PARAM + '/' + batch_size_path + lr_path + epoch_path + case_path
with open(param_dir, 'r') as f:
    TPAgent_kwargs = json.load(f)

# 에이전트 모델 로드
TPAgent = TargetPolicy(**TPAgent_kwargs)

# 학습 가중치 로드
weight_dir = parent_dir + '/weights/position-game/TargetPolicy'
weight_dir += '/' + batch_size_path + lr_path + epoch_path + case_path
weight_dir = weight_dir + '_mra={}_ro={}'.format(args.min_reward_action, args.reward_order)
print('TPAgent_weights_dir : ' , weight_dir)
TPAgent.load_weights(tf.train.latest_checkpoint(weight_dir))


'''
추론 루프
'''
# 메트릭 초기화
metrics_names = ['reward']


# 시간 초기화
total_start_time = time.time()

for epoch in range(1):
    start_time = time.time()
        
    for turn in range(epi_len - 1):

        # 행동 샘플링
        logits = TPAgent(test_inputs, training=False)                                               # logits : (num_epi, epi_len, action_size)
        sampled_actions = tf.random.categorical(logits=logits[:, -1, :], num_samples=1)             # sampled_actions : (num_epi, epi_len)      (stochastic sampling)
        # sampled_actions = tf.reshape(tf.argmax(logits[:, -1, :], axis=-1), shape = (-1, 1))             # sampled_actions : (num_epi, epi_len) (greedy sampling)
                                                                                                    # sampled_actions의 각 요소값들은 action 종류를 의미

        # 에피소드 스택
        sampled_actions_onehot = tf.one_hot(sampled_actions, depth = logits.shape[-1])      # sampled_actions_onehot : (num_epi, epi_len, action_size)
        # sampled_actions_onehot = test_inputs[:, -1, :][:, tf.newaxis, :] + sampled_actions_onehot
        test_inputs = tf.concat([test_inputs, sampled_actions_onehot], axis = 1)            # test_inputs : (num_epi, epi_len, action_size)

    # 보상 산출
    print('test_inputs.shape:', test_inputs[0, :, :])
    print('reward_dist.shape:', reward_dist)

    test_rewards = tf.squeeze(get_rewards(test_inputs, reward_dist))                     # test_rewards : (num_epi, epi_len, 1)
    print('test_reward :', test_rewards)

    print('\n')
    print('test_mean_rewards :', tf.reduce_mean(test_rewards).numpy())

# 케이스
print('case : ', kwargs['case'])
print('\n')

# test 행동 분포 저장
# test_action_dist = tf.cast(tf.reduce_sum(test_inputs[:, -1, :], axis = 0), dtype = tf.int32).numpy()
test_action_dist = tf.cast(tf.reduce_sum(tf.reshape(test_inputs, shape = (-1, logits.shape[-1])), axis = 0), dtype = tf.int32).numpy()
test_action_dist = tf.reshape(test_action_dist, shape = (1, -1))
action_list = np.arange(1, 11, 1).astype('str')
print('test_action_dist :', test_action_dist.shape)
test_action_dist_pd = pd.DataFrame(test_action_dist, columns = action_list)
file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
file_dir = file_dir + '_mra={}_ro={}'.format(args.min_reward_action, args.reward_order)
createFolder(file_dir)
file_name = '/test_action_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order)
test_action_dist_pd.to_csv(file_dir + file_name)
print('test_action_dist : ', test_action_dist)
print('\n')

# test 보상결과 저장
test_rewards_pd = pd.DataFrame(tf.squeeze(test_rewards), columns=np.arange(1, 11, 1))
file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
file_dir = file_dir + '_mra={}_ro={}'.format(args.min_reward_action, args.reward_order)
createFolder(file_dir)
file_name = '/test_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order)
test_rewards_pd.to_csv(file_dir + file_name)

# train 보상결과 및 행동 분포 출력
train_rewards = tf.squeeze(get_rewards(train_inputs[case], reward_dist))                     # train_rewards : (num_epi, epi_len, 1)
train_action_dist = tf.cast(tf.reduce_sum(train_inputs[case][:, -1, :], axis = 0), dtype = tf.int32)
print('train_mean_rewards :', tf.reduce_mean(train_rewards).numpy())
print('train_action_dist : ', train_action_dist)
print('\n')
