# %%
import os
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import get_params

'''
파라미터 로드
'''
args = get_params()
my_lr = args.rl_lr

'''
보상함수
'''
def reward_function(pred_logits: float, target_label: int):
    '''
    예측 logits 값을 받아서 보상으로 반환해주는 함수
    '''
    # logits을 probs로 변환
    pred_probs = tf.nn.softmax(pred_logits, axis = -1)

    # # logits (= probs)이 큰 값에 정답 (1), 작은 값에 오답 (0) 라벨을 부여
    # pred_labels = tf.argmax(pred_logits, axis = -1)

    # # 정오답 라벨을 one_hot 형태로 변환
    # num_labels = len(np.unique(pred_labels))
    # pred_onehot_labels = tf.one_hot(pred_labels, depth=num_labels)

    # # 정오답 라벨을 뒤집고 (1 - one_hot_label), 뒤집힌 값에 해당하는 (즉 오답에 해당하는) 확률을 보상으로 정의
    # # -- 오답이 제어하고자 하는 방향이므로, RL 입장에서는 정답임
    # rewards = tf.reduce_sum((1 - pred_onehot_labels) * pred_probs, axis = -1)[:, tf.newaxis]

    # target_label에 해당하는 확률값을 보상으로 정의
    rewards = pred_probs[:, target_label][:, tf.newaxis]

    return rewards      # rewards : (batch_size, 1)

def reward_dropout(rewards: float, dropout: str, dropout_rate: float):
    '''
    reward dropout 함수
    '''
    rewards = rewards.numpy()

    # random 드롭아웃
    if dropout == 'random':

        # dropout_rate 만큼의 샘플을 랜덤하게 뽑아, 해당 샘플들의 reward 드롭아웃
        batch_size = rewards.shape[0]
        dropout_size = int(batch_size * dropout_rate)
        dropout_idx = np.random.randint(batch_size, size=dropout_size)
        rewards[dropout_idx] = 0

    # quantile 드롭아웃
    elif dropout == 'quantile':

        # # 1-dropout_rate를 quantile로 간주하고, 해당 quantile 이하에 해당하는 샘플들의 reward 드롭아웃  (Quark의 아이디어 차용)
        # dropout_quantile = 1-dropout_rate
        # rewards_quantile = np.quantile(rewards, q=dropout_quantile)

        # dropout_rate를 quantile로 간주하고, 해당 quantile 이하에 해당하는 샘플들의 reward 드롭아웃  (Quark의 아이디어 차용)
        rewards_quantile = np.quantile(rewards, q=dropout_rate)
        rewards[rewards < rewards_quantile] = 0

    return rewards

'''
손실 함수 : CategoricalCrossEntropy (CCE)
'''
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
def loss_function(real, pred, mask):

    # 손실 계산
    losses = sparse_categorical_cross_entropy(real, pred)

    # 마스킹 적용
    mask = tf.cast(mask, dtype = losses.dtype)
    losses *= mask

    # 평균손실 계산
    # return tf.reduce_mean(losses)
    # return losses
    return tf.reduce_mean(losses, axis = -1)[:, tf.newaxis]

'''
정확도 함수
'''
def accuracy_function(real, pred, mask):
    real = tf.cast(real, dtype = tf.int32)

    # masking (= [MASK] + <pad>)되지 않은 토큰 갯수 세기 
    num_non_masks = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

    # masking (= [MASK] + <pad>)되지 않은 토큰 갯수 세기 
    non_mask_begin_idx = tf.argmax(mask, axis = -1)[0]

    # 최우도 토큰 반환
    max_pred = tf.argmax(pred, axis = -1)
    max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

    # 맞춘 토큰 행렬 (hit_matrix) 구축
    hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)

    if len(hit_index_mat) == 0:
        num_hits = 0
    else:
        # hit_matrix = tf.scatter_nd(hit_index_mat, np.repeat(1, hit_index_mat.shape[0]), shape = real.shape)
        hit_matrix = tf.scatter_nd(indices = hit_index_mat, updates = tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
        num_hits = tf.reduce_sum(hit_matrix[:, non_mask_begin_idx:], axis = -1)            

    # 각 sequence 별로 masking 되지않은 토큰들 중에서 맞춘 비율 계산
    acc = num_hits / num_non_masks
    mean_acc = tf.reduce_mean(acc)
    return tf.cast(mean_acc, dtype = tf.float32)
    # return tf.cast(acc, dtype = tf.float32)

'''
최적화 알고리즘 : Adam Optimizer
'''
optimizers = tf.keras.optimizers.Adam(learning_rate = my_lr)
# @tf.function
def control_step(data, model):

    inputs, inputs_mask, targets, targets_mask, rewards = data

    with tf.GradientTape() as tape:

        # 예측
        outputs = model(inputs, attention_mask = inputs_mask, training = True)

        # 손실 계산
        losses = loss_function(real=targets, pred=outputs.logits, mask=targets_mask)
        accuracies = accuracy_function(real=targets, pred=outputs.logits, mask=targets_mask)

        # 최종 손실
        total_losses = losses * rewards

    # 최적화
    gradients = tape.gradient(total_losses, model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.reduce_mean(losses), accuracies
    # return losses

# %%
'''
샘플링 함수
'''
class SamplingPortfolioOptimization:
    def __init__(self, **kwargs):
        self.decoding = kwargs['decoding']
        self.temperature = kwargs['temperature']
        self.top_k = kwargs['top_k']
        self.top_p = kwargs['top_p']
        self.portfolio_ratio = 1.0     # initial portfolio_ratio

    def portfolio_ratio_update(self, alpha:int=None, beta:int=None):
        new_portfolio_ratio = np.random.beta(a=alpha, b=beta)
        self.portfolio_ratio = new_portfolio_ratio

    def create_portfolio(self, logits):
    
        # sort tokens by their logits and define corresponding probabilities.
        sorted_index_vec = tf.argsort(logits[:, -1, :], axis = -1, direction='DESCENDING')
        sorted_logit_vec = tf.sort(logits[:, -1, :], axis = -1, direction='DESCENDING')
        # print('sorted_logit_vec : ', sorted_logit_vec)

        # vocab_size
        vocab_size = sorted_logit_vec.shape[1]

        # create "out-of" portfolio indices
        if self.portfolio_ratio < 1.0:

            # exponentiate and cumulate the values of sorted_logit_vec, which provides the cumulative probability density.
            sorted_prob_vec = tf.nn.softmax(sorted_logit_vec, axis = -1)
            cumul_sorted_prob_vec = tf.cumsum(sorted_prob_vec, axis = -1)

            # 샘플링 대상범위에 first_idx_greater_than_p까지 포함되어야 함. 다시 말해, 샘플링 제외범위에서는 포함되지 않아야 하므로 +1 붙여주기.
            first_idx_greater_than_p = tf.cast(tf.argmax(tf.math.greater(cumul_sorted_prob_vec, self.portfolio_ratio), axis = -1) + 1, dtype = tf.int32)
            mask_sequence = tf.ragged.range(first_idx_greater_than_p + 1, vocab_size)

            col_idx_vec = np.concatenate(list(mask_sequence))
            col_idx_vec = tf.cast(col_idx_vec, dtype = tf.int32)[:, tf.newaxis]

            row_idx_vec = np.concatenate([np.repeat(i, len(val)) for i, val in enumerate(mask_sequence)])
            row_idx_vec = tf.cast(row_idx_vec, dtype = tf.int32)[:, tf.newaxis]

            # concat row_idx vector and col_idx vector by column axis
            selected_idx = tf.concat([row_idx_vec,  col_idx_vec], axis = -1)

            # obtain sorted_portfolio_logits
            # add huge negative values (-1e+9) to "out-of" portfolio values.
            mask_matrix = tf.zeros(tf.shape(selected_idx)[0]) -1e+9             # mask_matrix는 -1e+9의 매우 큰 음수행렬이며, top_p에 해당하지 않은 부분을 채워줌.
            sorted_portfolio_logits = tf.tensor_scatter_nd_update(tensor=sorted_logit_vec, 
                                                                indices=selected_idx, 
                                                                updates=mask_matrix)
            
            # sorted_logit_vec_np = sorted_logit_vec.numpy()
            # target_value = np.quantile(sorted_logit_vec_np, q=1-self.portfolio_ratio, axis=1)
            # sorted_quantiled_logit_vec = sorted_logit_vec_np[sorted_logit_vec_np < target_value[:, np.newaxis]] = -1e+09

        else:
            sorted_portfolio_logits = sorted_logit_vec

        '''
        sorted_portfolio_logits을 원래 형태로 복원.
        '''
        batch_size = tf.shape(sorted_index_vec)[0]
        vocab_size = tf.shape(sorted_index_vec)[1]

        # 행 인덱스
        row_idx = tf.repeat(tf.range(batch_size), repeats=vocab_size)
        row_idx = row_idx[:, tf.newaxis]

        # 열 인덱스
        # double argsort trick을 통해 원래의 순서 복원
        # -- sorted_idx_vec은 이미 direction='DESCENDING'으로 내림차순 정렬되어 있음.
        # -- 아래의 second argsort()는 무조건 default direction인 오름차순 정렬해야 함 (double argsort trick의 원리임).
        col_idx = tf.argsort(sorted_index_vec, axis = -1)
        col_idx = tf.reshape(col_idx, shape = (-1, 1))

        # 최종 타겟 인덱스
        target_idx = tf.concat([row_idx, col_idx], axis = -1)

        # 복원
        portfolio_logits = tf.gather_nd(params=sorted_portfolio_logits, 
                                        indices=target_idx)
        portfolio_logits = tf.reshape(portfolio_logits, 
                                      shape = (tf.shape(sorted_portfolio_logits)[0], -1))

        return tf.reshape(portfolio_logits, shape=(batch_size, 1, -1))

    def sampling(self, logits, decoding:str, temperature:float, k:int, p:float):
        '''
        - max_new_tokens : 최대 토큰 샘플링 길이
        - decoding : 행동정책의 샘플링 (sampling) 유형
        - temperature : 온도 파라미터. 낮을수록 greedy decoding에 가까워지며 높을수록 high entropy 상태 (균일확률분포)가 됨.
        - k : top_k의 설정인자
        - p : top_p의 설정인자 (percentile의 약자)
        '''

        if decoding == 'greedy':
            preds = tf.argmax(logits[:, -1, :], axis = -1)[:, tf.newaxis]
            preds = tf.cast(preds, dtype = tf.int32)

        # 확률 샘플링
        elif decoding == 'stochastic':
            preds = tf.random.categorical(logits = logits[:, -1, :], num_samples = 1)
            preds = tf.cast(preds, dtype = tf.int32)

        # top-k 샘플링
        elif decoding == 'top_k':
            # get the top-k probability vector (*** 사실 top-k logits임)
            top_k_prob_vec = tf.math.top_k(logits[:, -1, :], k)[0]

            # normalize the probability mass by top-k tokens
            top_k_prob_vec_norm = tf.nn.softmax(top_k_prob_vec/temperature, axis = -1)
            
            # get the token vector of top-k probability
            top_k_token_vec = tf.math.top_k(logits[:, -1, :], k)[1]

            # sample the idx of logit vector according to logit value, which is a col_idx vector
            col_idx_vec = tf.cast(tf.random.categorical(logits = top_k_prob_vec_norm, num_samples = 1), dtype = tf.int32)
            
            # get the batch_size
            batch_size = tf.shape(top_k_prob_vec_norm)[0]

            # craete idx vector of batch_size, which is a the row_idx vector
            row_idx_vec = tf.reshape(tf.range(batch_size), shape = (-1, 1))

            # concat row_idx vector and col_idx vector by column axis
            selected_idx = tf.concat([row_idx_vec,  col_idx_vec], axis = -1)

            # obtain the values (tokens) from the top_k_token_vec defined above, which are top_k sampling tokens.
            preds = tf.reshape(tf.gather_nd(top_k_token_vec, indices=selected_idx), shape = (-1, 1))

        elif decoding == 'top_p':

            # sort tokens by their logits and define corresponding probabilities.
            sorted_token_vec = tf.argsort(logits[:, -1, :], axis = -1, direction='DESCENDING')
            sorted_logit_vec = tf.sort(logits[:, -1, :], axis = -1, direction='DESCENDING')

            # vocab_size
            vocab_size = sorted_logit_vec.shape[1]

            # create "out-of" top-p indices
            if p < 1.0:

                # exponentiate and cumulate the values of sorted_logit_vec, which provides the cumulative probability density.
                sorted_prob_vec = tf.nn.softmax(sorted_logit_vec, axis = -1)
                cumul_sorted_prob_vec = tf.cumsum(sorted_prob_vec, axis = -1)

                # 샘플링 대상범위에 first_idx_greater_than_p까지 포함되어야 함. 다시 말해, 샘플링 제외범위에서는 포함되지 않아야 하므로 +1 붙여주기.
                first_idx_greater_than_p = tf.cast(tf.argmax(tf.math.greater(cumul_sorted_prob_vec, p), axis = -1) + 1, dtype = tf.int32)
                mask_sequence = tf.ragged.range(first_idx_greater_than_p + 1, vocab_size)

                col_idx_vec = np.concatenate(list(mask_sequence))
                col_idx_vec = tf.cast(col_idx_vec, dtype = tf.int32)[:, tf.newaxis]

                # craete batch indices of "out-of" top-p indices
                # 각 샘플별로 first_idx_greater_than_p+1 이후의 남은 모든 단어들의 갯수만큼 샘플 인덱스를 repeat해주기.
                # row_idx_vec = tf.repeat(tf.range(0, batch_size), cumul_sorted_prob_vec.shape[1] - (first_idx_greater_than_p))[:, tf.newaxis]
                row_idx_vec = [np.repeat(i, len(val)) for i, val in enumerate(mask_sequence)]

                # concat row_idx vector and col_idx vector by column axis
                selected_idx = tf.concat([row_idx_vec,  col_idx_vec], axis = -1)

                # obtain top_p_logits
                # add huge negative values (-1e+9) to "out-of" top_p values.
                mask_matrix = tf.zeros(tf.shape(selected_idx)[0]) -1e+9             # mask_matrix는 -1e+9의 매우 큰 음수행렬이며, top_p에 해당하지 않은 부분을 채워줌.
                top_p_logits = tf.tensor_scatter_nd_update(tensor=sorted_logit_vec, indices=selected_idx, updates=mask_matrix)

            else:
                top_p_logits = sorted_logit_vec

            # sampling by top_p_logits and define it as col_idx2, which represents the top-p indices per sample
            col_idx_vec2 = tf.random.categorical(top_p_logits, num_samples=1, dtype = tf.int32)

            # define row_idx2, which represents the indices of samples
            row_idx_vec2 = tf.range(0, batch_size)[:, tf.newaxis]

            # finally, create a target idx matrix that indicates the probabilities of top-p sampled tokens.
            selected_idx2 = tf.concat([row_idx_vec2, col_idx_vec2], axis=-1)

            # based on the selected_idx2, gather the top-p sampled tokens.
            preds = tf.gather_nd(params=sorted_token_vec, indices=selected_idx2)[:, tf.newaxis]


        return preds

    # @tf.function
    def generation(self, inputs=None, attention_mask=None, model=None, max_new_tokens:int=None):
        '''
        - max_new_tokens : 최대 토큰 샘플링 길이
        '''

        gens = inputs.numpy()

        for i in range(max_new_tokens):

            outputs = model(gens, attention_mask=attention_mask, training=False)
            portfolio_logits = self.create_portfolio(logits=outputs.logits)

            preds = self.sampling(
                logits=portfolio_logits, 
                decoding=self.decoding,
                temperature=self.temperature,
                k=self.top_k,
                p=self.top_p
                )

            gens = tf.concat([gens, preds], axis = -1)
            attention_mask = tf.concat([attention_mask, tf.ones(shape=tf.shape(preds))], axis = -1)

        return gens

# %%
