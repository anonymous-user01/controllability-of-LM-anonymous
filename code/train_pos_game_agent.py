# %%
import os
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
import pandas as pd
import pickle, json
import copy
from model_agent import TargetPolicy
from utils import *
import time

'''
시드 설정
'''
os.environ["PYTHONHASHSEED"] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(47)
np.random.seed(47)
random.seed(47)

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

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
batch_size = args.batch_size
num_epoch = args.num_epoch
case = args.case

synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
kwargs = {
    'model_name' : 'TargetPolicy',
    'task' : 'position-game',
    'batch_size' : batch_size,
    'lr' : args.lr,
    'd_model' : 256,
    'd_embed' : 64,
    'num_layers' : 4,
    'action_size' : 10,
    'epi_len' : 10,
    'num_epoch' : num_epoch,
    'num_cases' : len(np.unique(synthetic_data['case']))
}

print('Case ranges in (0 - {}), and current case is : {}'.format(kwargs['num_cases']-1, case))

# kwargs = {
#     'model_name' : 'TargetPolicy',
#     'task' : 'position-game',
#     'batch_size' : 50,
#     'lr' : 1e-05,
#     'd_model' : 256,
#     'd_embed' : 64,
#     'num_layers' : 4,
#     'action_size' : 10,
#     'epi_len' : 10,
#     'num_epoch' : 1,
#     'early_stop' : 'yes',
#     'num_patience' : 5
# }

# 각종 경로 설정
model_name = str(kwargs['model_name'])
batch_size_path = str(kwargs['batch_size'])
lr_path = '_' + str(kwargs['lr'])
epoch_path = '_' + str(kwargs['num_epoch'])
case_path = '_' + str(case)

# 파라미터 셋팅 저장
SAVE_PATH_PARAM = set_SavePath(kwargs, save_file = 'params')    # 파라미터 저장 경로
param_dir = SAVE_PATH_PARAM + '/' + batch_size_path + lr_path + epoch_path + case_path
with open(param_dir, 'w') as f:
    json.dump(kwargs, f, ensure_ascii=False, indent = '\t')

'''
최적화 알고리즘, 손실함수 및 정확도 함수
'''
# 최적화 알고리즘
optimizers = tf.keras.optimizers.Adam(learning_rate = kwargs['lr'])

# 손실 함수
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
def loss_function(real, pred):
    '''
    real (= targets) : (batch_size, epi_len)
    pred (= logits) : (batch_size, epi_len, action_size)
    losses : (batch_size, epi_len)
    '''
    losses = sparse_categorical_cross_entropy(real, pred)
    return losses

# 정확도 함수
def acc_function(real, pred):
    real = tf.cast(real, dtype = tf.int32)

    # 예측 토큰 반환
    max_pred = tf.argmax(pred, axis = -1)
    max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

    # 맞춘 토큰 행렬 (hit_matrix) 구축
    hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)
    if len(hit_index_mat) == 0:
        num_hits = 0
    else:
        # hit_matrix = tf.scatter_nd(hit_index_mat, np.repeat(1, hit_index_mat.shape[0]), shape = real.shape)
        hit_matrix = tf.scatter_nd(hit_index_mat, tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
        num_hits = tf.reduce_sum(hit_matrix, axis = -1)            

    # padding 토큰 (token 0)에 대해서 masking된 행렬 구축
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    num_targets_without_padding = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

    # 각 sequence 별로 padding 제외 토큰들 중에서 맞춘 비율 계산
    acc = num_hits / num_targets_without_padding
    mean_acc = tf.reduce_mean(acc)
    return tf.cast(mean_acc, dtype = tf.float32)

'''
역전파 코드
'''
@tf.function
def train_step(data, model):
    '''
    inputs : (batch_size, epi_len, 10)
    targets : (batch_size, epi_len)
    rewards : (batch_size, epi_len)
    '''  
    inputs, targets, rewards = data

    with tf.GradientTape() as tape:

        # 예측
        logits = model(inputs, training = True)         # logits : (batch_size, epi_len, action_size)

        # 손실 및 정확도 계산
        losses = loss_function(targets, logits)         # losses : (batch_size, epi_len)
        accuracies = acc_function(targets, logits)

        # 중요도 샘플링 가중치 (분모를 생략한 근삿값) 계산
        probs = tf.math.softmax(logits, axis = -1)                      # probs : (batch_size, epi_len, action_size)
        onehots = tf.one_hot(targets, depth = inputs.shape[-1])         # onehots : (batch_size, epi_len, action_size)
        approx_IS_weights = tf.multiply(probs, onehots)                 # approx_IS_weights : (batch_size, epi_len, action_size)

        # 최종 손실
        IS_weighted_rewards = tf.multiply(tf.stop_gradient(approx_IS_weights), rewards[:, :, tf.newaxis])       # IS_weighted_rewards : (batch_size, epi_len, action_size)
        IS_weighted_rewards = tf.reduce_sum(IS_weighted_rewards, axis = -1)                                     # IS_weighted_rewards : (batch_size, epi_len)
        total_losses = IS_weighted_rewards * losses                                                             # total_losses : (batch_size, epi_len)
        # total_losses = (0 + tf.reduce_mean(IS_weighted_rewards)) * losses                                       # IS_weighted_rewards가 0이라도 behavior policy의 losses 업데이트는 수행되어야 하므로 + .001 해주기
        # total_losses = tf.reduce_mean(rewards) * losses

    # 최적화
    gradients = tape.gradient(total_losses, model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.reduce_mean(losses), accuracies


'''
데이터 로드
- train_inputs : cur_state를 정의하는 데이터 (num_cases, per_case_sample_size, epi_len, action_size)
- train_targets : next_state를 정의하는 action 데이터 (num_cases, per_case_sample_size, epi_len, 1)
- train_rewards : next_state에서 얻게되는 reward 데이터 (num_cases, per_case_sample_size, epi_len, 1)
'''
synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
reward_dist = pd.read_csv(parent_dir + '/prep_data/position-game/train_reward_dist_mra={}_ro={}.csv'.format(args.min_reward_action, args.reward_order), index_col=0)
reward_dist = np.concatenate(np.array(reward_dist))
train_inputs, train_targets, train_rewards = get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs)
sample_size = train_inputs[0].shape[0]

'''
저장 경로 생성
'''
SAVE_PATH_WEIGHT = set_SavePath(kwargs, save_file = 'weights')  # 학습 가중치 저장 경로
SAVE_PATH_RESULT = set_SavePath(kwargs, save_file = 'results')  # 결과 저장 경로

'''
RL 모델 정의
- state : [0, 0, ..., 0]의 10차원 벡터를 state로 정의. 
        : 각 element는 action의 index를 의미함
        : RL 훈련과정 동안 sampling한 action을 해당 action의 index와 일치한 element에 누적합산
        : 예) 만약 1번 action이 sampling 되면 [0, +1, 0, ...., 0], 거기서 또 0번 action이 샘플링되면 [+1, 1, 0, ..., 0]

- action : 1~10 사이의 숫자

- reward : synthetic데이터 생성시 적용햇던 option을 토대로 get_rewards(min_action, order)를 print하여 결정
        : 예를 들어, get_reward() 출력시 [0, 0, 0, 0, 0, 0.00022599, 0.00723164, 0.05491525, 0.23141243, 0.70621469]와 같은 vector가 반환된다면,
        : 1-5번 action은 reward가 없고, 6-10번 action은 위 출력 vector의 해당 index+1에 속하는 값을 reward로 반환하도록 설정
'''
TPAgent = TargetPolicy(**kwargs)

'''
데이터셋 구축
'''
with tf.device("/cpu:0"):

    # 학습 데이터
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs[case], train_targets[case], train_rewards[case]))
    train_batchset = train_dataset.batch(batch_size = kwargs['batch_size'], drop_remainder = True, num_parallel_calls = 8)
    train_batchset = train_batchset.prefetch(1)

'''
훈련 루프 수행
'''
# 메트릭 초기화
metrics_names = [str(model_name) + '_loss', str(model_name) + '_acc']
train_loss_history = []
train_acc_history = []

# 시간 초기화
total_start_time = time.time()

for epoch in range(kwargs['num_epoch']):
    start_time = time.time()

    # 매 epoch 마다 누적 손실 및 정확도 초기화
    train_cumul_acc = train_cumul_loss = 0

    # 매 epoch 마다 진행상태바 초기화
    print("\nepoch {}/{}".format(epoch + 1, kwargs['num_epoch']))
    pb_i = Progbar(len(train_batchset), stateful_metrics = metrics_names)

    '''
    학습 배치 루프
    '''

    # 훈련 배치 루프
    for idx, (train_inputs, train_targets, train_rewards) in enumerate(train_batchset):

        # train_targets = tf.squeeze(train_targets)
        # train_rewards = tf.squeeze(train_rewards)
        train_targets = tf.reshape(train_targets, shape = (kwargs['batch_size'], -1))
        train_rewards = tf.reshape(train_rewards, shape = (kwargs['batch_size'], -1))

        # train_reward_np = train_rewards.numpy() + 1.0
        # # random_idx = np.random.randint(low=0, high=train_reward_np.shape[0], size = int(train_reward_np.shape[0] * 0.85))
        # # reward_q = np.quantile(train_rewards.numpy(), q = 0.01)
        # # print('reward_q : {}'.format(reward_q))
        # # random_idx = np.where(train_rewards.numpy() <= reward_q)[0]
        # # random_idx = np.where(train_rewards.numpy() <= 0.0000001)[0]
        # # train_reward_np[[random_idx]] = 0
        # train_rewards = tf.cast(train_reward_np, dtype=tf.float32)

        # 손실 및 정확도 산출 (순전파 및 역전파 수행)
        train_loss, train_acc = train_step((train_inputs, train_targets, train_rewards), model=TPAgent)

        # 배치별 손실 및 정확도 누계
        train_cumul_loss += train_loss.numpy()
        train_cumul_acc += train_acc.numpy()

        # 메트릭 값 업데이트
        metric_values = [(str(model_name) + '_loss', train_loss), (str(model_name) + '_acc', train_acc)]
        pb_i.update(idx+1, values = metric_values)


    # 전체 평균 손실 및 정확도 (훈련셋)
    train_mean_loss = train_cumul_loss/(idx + 1)
    train_mean_acc = train_cumul_acc/(idx + 1)

    # 훈련 성능 출력
    print('train_mean_loss : {}, train_mean_acc : {}'.format(train_mean_loss, train_mean_acc))

    # 가중치 저장 조건
    # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
    weight_dir = SAVE_PATH_WEIGHT + '/' + batch_size_path + lr_path + epoch_path + case_path
    weight_dir = weight_dir + '_mra={}_ro={}'.format(args.min_reward_action, args.reward_order)
    createFolder(weight_dir)
    TPAgent.save_weights(weight_dir + '/weights.ckpt')


    # 훈련 셋 손실 히스토리 저장
    train_loss_history += [train_mean_loss]
    train_acc_history += [train_mean_acc]
    loss_acc_history_pd = pd.DataFrame(zip(train_loss_history, train_acc_history), columns = ['train_loss', 'train_acc'])
    file_dir = SAVE_PATH_RESULT + '/' + batch_size_path + lr_path + epoch_path + case_path
    file_dir = file_dir + '_mra={}_ro={}'.format(args.min_reward_action, args.reward_order)
    createFolder(file_dir)
    file_name = '/loss_acc_history.csv'
    loss_acc_history_pd.to_csv(file_dir + file_name, index_label = 'epoch')


    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))
