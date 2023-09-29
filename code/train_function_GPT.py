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
my_lr = args.lr

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
    return tf.reduce_mean(losses)

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

'''
최적화 알고리즘 : Adam Optimizer
'''
optimizers = tf.keras.optimizers.Adam(learning_rate = my_lr)
@tf.function
def train_step(data, model):

    inputs, inputs_mask, targets, targets_mask = data

    with tf.GradientTape() as tape:

        # 예측
        outputs = model(inputs, attention_mask = inputs_mask, training = True)

        # 손실 계산
        losses = loss_function(real=targets, pred=outputs.logits, mask=targets_mask)
        accuracies = accuracy_function(real=targets, pred=outputs.logits, mask=targets_mask)

        # 최종 손실
        total_losses = losses

    # 최적화
    gradients = tape.gradient(total_losses, model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))

    return losses, accuracies

@tf.function
def test_step(data, model):

    inputs, inputs_mask, targets, targets_mask = data

    # 예측
    outputs = model(inputs, attention_mask = inputs_mask, training = False)

    # 손실 및 정확도 계산
    losses = loss_function(real=targets, pred=outputs.logits, mask=targets_mask)
    accuracies = accuracy_function(real=targets, pred=outputs.logits, mask=targets_mask)

    return losses, accuracies