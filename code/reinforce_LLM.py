# %%
import os
from pathlib import Path
import sys
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import time
import tqdm
import glob

from utils import get_params, set_save_dir, truncate_datasize_by_ratio, get_truncated_data, indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token
from tensorflow.keras.utils import Progbar
from load_pretrained_LLM import bert_tokenizer, bert_model, gpt_tokenizer, gpt_model, target_model
from train_function_RL import reward_function, reward_dropout, control_step

'''
파라미터 로드
'''
args = get_params()
my_task = args.task
my_dataset = args.dataset
my_model = args.model
my_rl_lr = args.rl_lr
my_rl_bs = args.rl_batch_size
my_rl_epoch = args.rl_num_epoch
my_decoding = args.decoding        # decodnig strategy
my_prefix_len = args.prefix_len    # length of prefix to use
my_gen_len = args.gen_len          # length of generation
my_dropout = args.dropout
my_dropout_rate = args.dropout_rate

parent_dir = str(Path(os.getcwd()).parents[0])
PARAM_DIR = parent_dir + '/params' + '/' + str(my_dataset) + '/' + str(my_model)
PARAM_DIR = glob.glob(PARAM_DIR + '/*{}**{}**{}**{}**{}**{}**{}**{}**{}*'.format(my_task, my_rl_lr, my_rl_bs, my_rl_epoch, my_decoding, my_prefix_len, my_gen_len, my_dropout, my_dropout_rate))[0]
with open(PARAM_DIR + '/kwargs.json', 'r') as f:
    kwargs = json.load(f)

'''
데이터 로드
'''
# 데이터 로드 경로 설정
prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/' + my_model

'''
훈련용 데이터 셋팅
'''
# 훈련용 인풋 시퀀스 및 어텐션 마스크 로드
train_input_x = np.load(prep_data_path + '/train_input_x.npy')
train_input_att = np.load(prep_data_path + '/train_input_att.npy')
# train_y = np.load(prep_data_path + '/train_y.npy')

'''
주어진 프롬프트 길이 (prefix_len) 내에 <pad>가 존재하는지 여부 확인 및 <pad> 존재 시퀀스 삭제
- 앞서 preprocessing 단계에서 left-padding으로 토크나이징 할 때, 최대 길이를 30으로 제한하였음. 따라서, 길이가 30 미만인 문장들에는 prefix_len 내에 <pad>가 반드시 존재함
- 다시 말해, "prefix_len 내에 <pad>가 존재하는 시퀀스를 삭제한다 = 길이가 30 미만인 문장들 삭제한다" 가 된다.
- 그러나 본질적인 의미는, left-padding에 의해 prefix_len 내에 <pad>가 존재할 경우, <pad>라는 prefix로부터 이어서 문장을 생성하는것은 불가능하므로, 이를 방지하기 위함이다.
'''
# --> decoder-only architecture는 <pad>가 앞에서부터 존재하는데,
# --> reconstruction 포맷으로 (즉, input = output = target (right-shift) 데이터 포맷으로) 학습데이턱 입력되는 경우,
# --> input의 앞쪽에 등장하는 <pad>가 output (및 target)의 앞쪽에도 동일하게 등장하여,
# --> 결과적으로 concat(input, output) = input_seq의 중간에 <pad>가 존재하며, 
# --> decoder-only architecture에 맞지 않는다는 에러를 유발함.
target_idx = indice_pad_in_prefix(prefix_data=train_input_x, prefix_len=my_prefix_len, pad_token_id=gpt_tokenizer.pad_token_id)
train_input_x = remove_pad_in_prefix_case(target_idx=target_idx, target_data=train_input_x)
train_input_x = tf.cast(train_input_x, dtype=tf.int32)
train_input_att = remove_pad_in_prefix_case(target_idx=target_idx, target_data=train_input_att)
train_input_att = tf.cast(train_input_att, dtype=tf.float32)


'''
데이터 서브 샘플링
- emotion 데이터 일 때는 이 코드 주석처리하고 돌리기 (즉, emotion 데이터의 경우 데이터 양 자체가 적은데다 label class가 여러개임으로 모든 데이터 전부 활용하도록 설정)
'''
if my_dataset.split('-')[0] == 'emotion' or my_dataset.split('-')[0] == 'act':
    '''
    emtion & act 데이터 사이즈 : (76053)
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.7)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'politeness':
    '''
    politeness 데이터 사이즈 : (1121980, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.05)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'sentiment':
    '''
    sentiment 데이터 사이즈 : (560000, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.1)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'topic':
    '''
    topic 데이터 사이즈 : (120000, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.5)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'toxicity':
    '''
    toxicity 데이터 사이즈 : (159571, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.4)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

'''
Prefix 자르기
'''
train_input_x_prefixed = train_input_x[:, :my_prefix_len]
train_input_att_prefixed = train_input_att[:, :my_prefix_len]

'''
훈련 데이터셋 구축
'''
with tf.device("/cpu:0"):

    # 훈련 셋
    train_dat = tf.data.Dataset.from_tensor_slices((train_input_x_prefixed, train_input_att_prefixed, train_input_x)).shuffle(buffer_size = train_input_x_prefixed.shape[0], reshuffle_each_iteration = False)
    train_batch = train_dat.batch(batch_size=my_rl_bs, drop_remainder=True)

'''
훈련 과정 수행
'''
# 학습 가중치 및 결과 저장경로 생성
SAVE_WEIGHT_DIR = set_save_dir(kwargs, folder='weights', subfolder=my_dataset + '/' + my_model)
SAVE_RESULT_DIR = set_save_dir(kwargs, folder='results', subfolder=my_dataset + '/' + my_model)

# 훈련 메트릭
metrics_names = [str(my_model) + '_loss', str(my_model) + '_acc', str(my_model) + '_reward']

# 그 외
train_loss_history = []
train_acc_history = []
train_reward_history = []
target_label = int(my_dataset.split('-')[-1])

# 훈련 루프
total_start_time = time.time()
for epoch in range(my_rl_epoch):

    # 훈련 시간 및 진행상황 측정
    start_time = time.time()
    print("\nepoch {}/{}".format(epoch + 1, my_rl_epoch))
    pb_i = Progbar(len(train_batch), stateful_metrics = metrics_names)

    # 매 epoch마다 훈련 정확도, 손실 및 보상 초기화
    train_cumul_acc = 0
    train_cumul_loss = 0
    train_cumul_reward = 0

    # 훈련 배치 루프
    for idx, (train_input_x_prefixed, train_input_att_prefixed, train_input_x) in enumerate(train_batch):        

        '''
        1) 타겟 모델 train
        '''
        # 시퀀스 생성
        if my_decoding == 'greedy':
            bp_gen_texts = gpt_model.generate(train_input_x_prefixed, attention_mask = train_input_att_prefixed, 
                                                max_new_tokens=my_gen_len, 
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=False)
        elif my_decoding == 'stochastic':
            bp_gen_texts = gpt_model.generate(train_input_x_prefixed, attention_mask = train_input_att_prefixed, 
                                                max_new_tokens=my_gen_len, 
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, temperature=1.0)
        elif my_decoding == 'top-k':
            bp_gen_texts = gpt_model.generate(train_input_x_prefixed, attention_mask = train_input_att_prefixed, 
                                                max_new_tokens=my_gen_len, 
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, top_k=10, temperature=1.0)
        elif my_decoding == 'top-p':
            bp_gen_texts = gpt_model.generate(train_input_x_prefixed, attention_mask = train_input_att_prefixed, 
                                                max_new_tokens=my_gen_len, 
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, top_p=0.95, top_k=0, temperature=1.0)
        elif my_decoding == 'beam':
            bp_gen_texts = gpt_model.generate(train_input_x_prefixed, attention_mask = train_input_att_prefixed, 
                                                max_new_tokens=my_gen_len, 
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                num_beams=3, repetition_penalty=1.2, early_stopping=True)

        # 생성된 시퀀스의 보상 계산
        bp_padded_gen_texts = right_pad_after_eos_token(
                                    bp_gen_texts, 
                                    eos_token_id=gpt_tokenizer.eos_token_id, 
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    total_len=my_gen_len
                                    )
        
        bp_gen_texts_decoded = gpt_tokenizer.batch_decode(bp_padded_gen_texts)   # 보상 계산은 생성이 시작된 <bos> 이후부터 계산해야 하므로 "+1" 적용
        bp_gen_texts_bert_encoded = bert_tokenizer(bp_gen_texts_decoded, return_tensors='np', truncation=True, max_length=my_gen_len, padding=True)     # <bos> 포함
        bert_bp_gen_texts = bp_gen_texts_bert_encoded['input_ids']
        bert_bp_gen_masks = bp_gen_texts_bert_encoded['attention_mask']        
        bp_pred_logits = bert_model(bert_bp_gen_texts, attention_mask = bert_bp_gen_masks, training = False)
        
        # 드롭아웃 적용시
        if my_dropout != "None":
            bp_rewards = reward_function(bp_pred_logits, target_label)
            bp_rewards = reward_dropout(bp_rewards, dropout=my_dropout, dropout_rate=my_dropout_rate)
        else:
            bp_rewards = reward_function(bp_pred_logits, target_label)
    
        # 생성된 시퀀스로 target_policy_model을 훈련시킬 input, output, target (= right-shifted output) 데이터 만들기
        rl_input_seq = bp_padded_gen_texts[:, :-1]            # bp_padded_gen_texts[:, :-1] : bp_gen_texts[:, input_len:-1] 가 right_pad 된 상태
        rl_target_seq = bp_padded_gen_texts[:, 1:]            # bp_padded_gen_texts[:, 1:] : bp_gen_texts[:, input_len+1:] 가 right_pad 된 상태

        # attention mask 데이터 만들기
        rl_input_mask = tf.math.not_equal(rl_input_seq, gpt_tokenizer.pad_token_id)
        rl_target_mask = tf.math.not_equal(rl_target_seq, gpt_tokenizer.pad_token_id)

        # 강화학습
        train_loss, train_acc = control_step((rl_input_seq, rl_input_mask, rl_target_seq, rl_target_mask, bp_rewards), target_model)


        '''
        2) 타겟 모델 inference 및 보상 계산
        - 타겟 모델 생성결과를 출력할 때는 그리디 디코딩으로 생성해야 함.
        - 이는 reward의 확실한 상승 추세를 보기 위함임.
        - eval() 단계에서 생성을 할 때는 stochastic decoding으로 생성해도 됨.
        '''
        tp_gen_texts = target_model.generate(train_input_x_prefixed,
                                                attention_mask = train_input_att_prefixed,
                                                max_new_tokens=my_gen_len,
                                                pad_token_id=gpt_tokenizer.pad_token_id,
                                                eos_token_id=gpt_tokenizer.eos_token_id,
                                                do_sample=False)
        padded_tp_gen_texts = right_pad_after_eos_token(
                                    tp_gen_texts, 
                                    eos_token_id=gpt_tokenizer.eos_token_id, 
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    total_len=my_gen_len
                                    )
        tp_gen_texts_decoded = gpt_tokenizer.batch_decode(padded_tp_gen_texts)   # 보상 계산은 생성이 시작된 <bos> 이후부터 계산해야 하므로 "+1" 적용
        tp_gen_texts_bert_encoded = bert_tokenizer(tp_gen_texts_decoded, return_tensors='np', truncation=True, max_length=my_gen_len, padding=True)     # <bos> 포함
        bert_tp_gen_texts = tp_gen_texts_bert_encoded['input_ids']
        bert_tp_gen_masks = tp_gen_texts_bert_encoded['attention_mask']        
        tp_pred_logits = bert_model(bert_tp_gen_texts, attention_mask = bert_tp_gen_masks, training = False)
        tp_rewards = reward_function(tp_pred_logits, target_label)

        '''
        3) 배치별 학습 현황 모니터링 (메트릭 업데이트)
        '''
        # 메트릭 값 업데이트
        metric_values = [(str(my_model) + '_loss', train_loss), (str(my_model) + '_acc', train_acc), (str(my_model) + '_reward', tf.reduce_mean(tp_rewards).numpy())]
        pb_i.update(idx+1, values = metric_values)

        # 배치별 정확도, 손실 및 보상 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()
        train_cumul_reward += tf.reduce_mean(tp_rewards).numpy()

        '''
        4) 중간중간 출력하기
        '''
        if idx % 10 == 0:
            print('\n')
            print('original texts : {}'.format(gpt_tokenizer.batch_decode(train_input_x[:1, 1:])))
            print('\n')
            print('prefix : {}'.format(gpt_tokenizer.batch_decode(train_input_x_prefixed[:1, 1:])))
            print('\n')
            print('gen texts : {}'.format(gpt_tokenizer.batch_decode(padded_tp_gen_texts[:1, 1:])))


    # 전체 평균 정확도, 손실 및 보상 (훈련셋)
    train_mean_acc = train_cumul_acc/(idx + 1)
    train_mean_loss = train_cumul_loss/(idx + 1)
    train_mean_reward = train_cumul_reward/(idx + 1)

    # 훈련 성능 출력
    print('train_mean_loss : {}, train_mean_acc : {}, train_mean_reward : {}'.format(train_mean_loss, train_mean_acc, train_mean_reward))

    # 가중치 저장 조건
    '''
    test set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
    '''
    # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
    target_model.save_weights(SAVE_WEIGHT_DIR + '/epoch={}_weights.ckpt'.format(epoch))

    # 훈련 손실 히스토리 저장
    train_loss_history += [train_mean_loss]
    loss_history_pd = pd.DataFrame(train_loss_history, columns = ['train_loss'])
    loss_history_pd.to_csv(SAVE_RESULT_DIR + '/loss_history.csv', index_label = 'epoch')

    # 훈련 정확도 히스토리 저장
    train_acc_history += [train_mean_acc]
    acc_history_pd = pd.DataFrame(train_acc_history, columns = ['train_acc'])
    acc_history_pd.to_csv(SAVE_RESULT_DIR + '/acc_history.csv', index_label = 'epoch')

    # 훈련 보상 히스토리 저장
    train_reward_history += [train_mean_reward]
    reward_history_pd = pd.DataFrame(train_reward_history, columns = ['train_reward'])
    reward_history_pd.to_csv(SAVE_RESULT_DIR + '/reward_history.csv', index_label = 'epoch')

    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))
