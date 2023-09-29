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
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
def loss_function(real, pred):
    # 손실 계산
    losses = sparse_categorical_cross_entropy(real, pred)

    # 평균손실 계산
    return tf.reduce_mean(losses)


'''
정확도 함수
'''
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_categorical_accuracy')
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_categorical_accuracy')
def accuracy_function(real, pred, mode):
    
    if mode == 'train':
        # train_acc_metric.update_state(real, pred)
        # mean_acc = train_acc_metric.result()
        mean_acc = train_acc_metric(real, pred)
    elif mode == 'test':
        # val_acc_metric.update_state(real, pred)
        # mean_acc = val_acc_metric.result()
        mean_acc = train_acc_metric(real, pred)

    return tf.cast(mean_acc, dtype = tf.float32)


'''
최적화 알고리즘 : Adam Optimizer
'''
optimizers = tf.keras.optimizers.Adam(learning_rate = my_lr)    

@tf.function
def train_step(data, model):
    '''
    model = BERT
    '''

    inputs, masks, labels = data

    with tf.GradientTape() as tape:
        # 예측
        label_preds = model(inputs, attention_mask = masks, training = True)

        # 손실 및 정확도 계산
        losses = loss_function(real=labels, pred=label_preds)

        # 최종 손실
        total_losses = losses

    # 최적화
    gradients = tape.gradient(total_losses, model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))
    accuracies = accuracy_function(labels, label_preds, mode = 'train')

    return losses, accuracies

@tf.function
def test_step(data, model):
    '''
    model = BERT
    '''
    inputs, masks, labels = data

    # 예측
    label_preds = model(inputs, attention_mask = masks, training = False)

    # 손실 및 정확도 계산
    losses = loss_function(real=labels, pred=label_preds)
    accuracies = accuracy_function(real=labels, pred=label_preds, mode = 'test')

    return losses, accuracies
