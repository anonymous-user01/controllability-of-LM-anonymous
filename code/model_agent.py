# %%
from operator import concat
import sys
from attr import attrib
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import copy
from utils import *
from tqdm import tqdm

'''
Target Policy 모델
'''
# class TargetPolicy(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super(TargetPolicy, self).__init__()

#         # 모델 관련 공유 파라미터
#         self.batch_size = kwargs['batch_size']
#         # self.mask_generator = Mask_Generator()

#         # 디코더 파라미터
#         self.decoder = Decoder(**kwargs)

#     def call(self, inputs):

#         # 인풋 데이터 및 패딩토큰 마스킹 행렬 준비
#         _, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, inputs)

#         # 서브시퀀스 패딩 무효화
#         dec_subseq_mask = tf.ones(shape=dec_subseq_mask.shape)

#         # Decoder 네트워크
#         logits, att_weights = self.decoder(inputs, dec_pad_mask, dec_subseq_mask)      # decoder_outputs : (batch_size, seq_len, voca_size)

#         return logits, att_weights

class TargetPolicy(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TargetPolicy, self).__init__()

        # 모델 관련 공유 파라미터
        self.batch_size = kwargs['batch_size']
        self.action_size = kwargs['action_size']
        self.d_embed = kwargs['d_embed']
        self.d_model = kwargs['d_model']
        self.num_layers = kwargs['num_layers'] - 1      # self.embed_to_layer가 num_layers에 count 되니까 -1 해주기.

        # 레이어
        self.embedding_layer = tf.keras.layers.Dense(units=self.d_embed)
        self.embed_to_linear = tf.keras.layers.Dense(units=self.d_model)
        self.linear_layer = tf.keras.layers.Dense(units=self.d_model)
        self.layer_stack = [self.embed_to_linear] + [self.linear_layer for i in range(self.num_layers)]
        self.output_layer = tf.keras.layers.Dense(units=self.action_size)

    def call(self, inputs):

        # 임베딩 레이어
        embeds = self.embedding_layer(inputs)               # embeds : (batch_size, epi_len, d_embed)

        # 압축 레이어
        for i, a_layer in enumerate(self.layer_stack):
            if i == 0:
                outputs = a_layer(embeds)                   # outputs : (batch_size, epi_len, d_model)
            else:
                outputs = a_layer(outputs)                  # outputs : (batch_size, epi_len, d_model)

        # 최종 아웃풋
        final_outputs = self.output_layer(outputs)          # final_outputs : (batch_size, epi_len, action_size)

        return final_outputs