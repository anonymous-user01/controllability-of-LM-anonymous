# %%
import os
from pathlib import Path
import sys
import argparse
import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
초기화, 시드지정, 에러표시 함수
'''
def initialize_setting():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def get_params():
    parser = argparse.ArgumentParser(description='receive the parameters')

    # 필수 인자들 설정
    parser.add_argument('--my_seed', type = int, required = True)
    parser.add_argument('--model', type = str, required = True)         # model : {'gpt2_small', 'gpt2_large', 'dialoGPT', 'bert'}
    parser.add_argument('--task', type = str, required = True)          # task : {'ft', 'rl', 'train_eval', 'test_eval'} , ft = fine-tuning, rl = reinforcement-learning
    parser.add_argument('--dataset', type = str, required = True)       # dataset = {'sentiment-0', 'sentiment-1', 
                                                                                    # 'toxicity-0', 'toxicity-1', 
                                                                                    # 'politeness-0', 'politeness-1',
                                                                                    # 'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                                    # 'act-0', 'act-1', 'act-2', 'act-3', 'act-4'}

    parser.add_argument('--batch_size', type = int, required = False)
    parser.add_argument('--lr', type = float, required = False)
    parser.add_argument('--num_epoch', type = int, required = False)
    parser.add_argument('--num_patience', type = int, required = False)

    parser.add_argument('--rl_batch_size', type = int, required = False)        # rl_batch_size : 임의의 정수
    parser.add_argument('--rl_lr', type = float, required = False)              # rl_lr : 임의의 실수
    parser.add_argument('--rl_num_epoch', type = int, required = False)         # rl_num_epoch : 임의의 정수
    parser.add_argument('--decoding', type = str, required = False)             # decoding : {'greedy', 'stochastic', 'top-k', 'top-p', 'beam'}
    parser.add_argument('--prefix_len', type = int, required = False)           # prefix_len : 임의의 정수
    parser.add_argument('--gen_len', type = int, required = False)              # gen_len : 임의의 정수
    parser.add_argument('--dropout', type = str, required = False)              # dropout : 드롭아웃 스타일 {'random', 'quantile'}
    parser.add_argument('--dropout_rate', type = float, required = False)       # dropout_rate : 드롭아웃 비율, 임의의 [0, 1]

    parser.add_argument('--test_prefix', type=str, required=False)              # test_prefix : evaluate_LLM 단계에서 초기 값으로 넣어주는 단어
    global args
    args = parser.parse_args()

    return args

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class WindowLengthError(Exception):
    def __str__(self):
        return print("Error! Enter right option for \"dec_window_masking\" parameter. \n You must enter an odd number for \'dec_masking_window_len\'.")

'''
파일명이 name과 완벽하게 matching 되는 파일을 path의 모든 하위 디렉토리에서 검색
'''
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

'''
폴더 생성
'''
def createFolder(directory: str):
    try:
        if os.path.exists(directory) == False:
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


'''
경로 지정 및 생성
'''
def set_SavePath(kwargs, save_file = 'weights'):

    # save_file 및 save_model에 따라 파일저장 경로 별도로 생성
    save_task = '/' + kwargs['task']
    save_model = '/' + kwargs['model_name']
    SAVE_PATH = parent_dir + '/' + save_file + save_task + save_model

    # SAVE_PATH 경로가 존재하지 않는 경우에 해당 경로를 생성
    createFolder(SAVE_PATH)

    return SAVE_PATH

def name_save_file(kwargs: dict) -> str:
    return ''.join([ '_' + str(val) if idx > 0 else str(val) for idx, val in enumerate(kwargs.values())])

# def name_save_file(kwargs: dict) -> str:
#     temp = []
#     for idx, val in enumerate(kwargs.items()):
#         if val[0] in ['dataset', 'decoding', 'dropout', 'dropout_rate']:
#             temp += [val[1]]

#     return ''.join([ '_' + str(val) if idx > 0 else str(val) for idx, val in enumerate(temp)])

def set_save_dir(kwargs: dict, folder=None, subfolder=None) -> str:

    save_dir = parent_dir + '' + '/' + folder + '/' + subfolder
    save_file_name = name_save_file(kwargs)
    save_file_dir = save_dir + '/' +  save_file_name
    createFolder(save_file_dir)

    return save_file_dir


'''
agnews 데이터 csv 파일로 저장
'''
def agnews_to_csv(data, split:str):
    '''
    data : load_data('ag_news')로 내려받은 파일의 .data 
    split : 'train' 또는 'test' ('val'이나 'dev'가 있을 수도 있음)
    '''

    # 데이터 변환
    data_df = pd.DataFrame(list(data[split]))
    data_df = data_df.T
    data_df.columns = ['text', 'label']

    # 저장
    data_df.to_csv(parent_dir + '/data/topic/{}.csv'.format(split))

    return data_df

'''
daily dailog 데이터 전처리 및 저장
'''
def preprocess_n_save_DailyDialog(daily_dialog_raw_dataset, split: str, N_turn: int = None):
    '''
    daily_dialog_raw_dataset : load_dataset('daily_dialog')
    - sentence_dialog : n개의 sentence 묶음인 dialog 단위의 데이터를 sentence로 분해한 데이터
    - dialog_idx : 각 sentence가 어떤 dialog에 속하는지 나타내는 인덱스
    - sentence_dialog_with_index : 각 sentence가 어떤 dialog에 속하는지 인덱스를 달아놓은 데이터
    - full_dataset : dialog, action, emotion 관련 column이 모두 존재하는 데이터셋 
    - cum_sentence_dialog : 각 인덱스 별, 즉 각 dialog 내에서 발생한 sentence들을 순서대로 누적시킨 데이터
    - indexBag : ㅇㅇㅇㅇ
    '''

    dialog_set = daily_dialog_raw_dataset.data[split]['dialog']

    # 대화를 문장 단위로 쪼개어 데이터셋을 구성하고, 대화를 그릅화하는 인덱스 생성
    sentence_dialog = pd.DataFrame([sentence for dialog in dialog_set for sentence in dialog ], columns = ['query'])
    dialog_idx = pd.DataFrame(np.concatenate([np.repeat(idx, len(dialog)) for idx, dialog in enumerate(dialog_set)]), columns=['dialog_idx'])
    sentence_dialog_with_index = pd.concat([dialog_idx, sentence_dialog], axis = 1)

    # 문장단위로 쪼개진 데이터셋에 맞추어 act 및 emotion 컬럼을 구축하고, 하나의 데이터 프레임 full_dataset 으로 정의
    act_list_set = daily_dialog_raw_dataset.data[split]['act']
    act_vec = pd.DataFrame([act for act_list in act_list_set for act in act_list ], columns = ['act_q'])
    full_dataset = pd.concat([sentence_dialog_with_index, act_vec], axis = 1)

    emo_list_set = daily_dialog_raw_dataset.data[split]['emotion']
    emo_vec = pd.DataFrame([emotion for emo_list in emo_list_set for emotion in emo_list ], columns = ['emo_q'])
    full_dataset = pd.concat([full_dataset, emo_vec], axis = 1)

    # 대화의 누적응답 기록 생성
    cum_sentence_dialog, indexBag, all_responseBag, all_act_r_Bag, all_emo_r_Bag = cumulate_sentence_dialog(full_dataset)
    cum_sentence_dialog = cum_sentence_dialog['dialog_history'].apply(lambda x : '<\s>'.join(x))

    # 누적응답 (dialog_history), 질의 (query), 응답 (response) triplet에 맞춘 act_q, emo_q, act_r, emo_r 컬럼을 구축하고, 하나의 데이터 프레임 indiced_full_dataset 으로 정의
    indiced_full_dataset = full_dataset.iloc[indexBag]
    indiced_full_dataset.insert(loc=1, value=list(cum_sentence_dialog), column='dialog_hist')
    indiced_full_dataset.insert(loc=3, value=list(all_responseBag), column='response')
    indiced_full_dataset = pd.concat([indiced_full_dataset, all_act_r_Bag.set_index(indiced_full_dataset.index)], axis = 1)
    indiced_full_dataset = pd.concat([indiced_full_dataset, all_emo_r_Bag.set_index(indiced_full_dataset.index)], axis = 1)

    # 대화 Turn의 갯수가 N_turn개 이하인 대화로만 구성하기
    if N_turn != None:
        N_turn_dialog_idx = np.where(np.unique(indiced_full_dataset['dialog_idx'], return_counts=True)[1] <= N_turn)[0]
        indiced_full_dataset = indiced_full_dataset[indiced_full_dataset['dialog_idx'].isin(N_turn_dialog_idx) == 1]

    # 최종 데이터셋 (indiced_full_dataset) 저장
    indiced_full_dataset.to_csv(parent_dir + '/data/dialog/' + split + '_{}.csv'.format(N_turn))

    # 대화의 평균길이
    mean_dialog_len = np.mean(indiced_full_dataset.groupby('dialog_idx').apply(lambda x : len(x)+1))
    print('{}_dialog mean_len : {}'.format(split, mean_dialog_len))

    return sentence_dialog_with_index, full_dataset, cum_sentence_dialog, indiced_full_dataset

'''
대화기록 누적 함수
'''
def cumulate_sentence_dialog(full_dataset):
    '''
    num_dialog_set : 총 dialog_set 갯수
    historyBag : 각 dialog 마다 진행되는 sentence를 누적시켜 담아놓은 bag 
    '''
    num_dialog_set = len(np.unique(full_dataset['dialog_idx']))
    historyBag = []
    indexBag = []
    all_responseBag = []
    all_act_r_Bag = []
    all_emo_r_Bag = []
    for dialog_idx in range(num_dialog_set):
        a_dialog = full_dataset['query'][full_dataset['dialog_idx'] == dialog_idx]

        queryBag = []
        cum_queryBag = []
        responseBag = []

        # 대화별 누적응답 기록 (responseBag) 추출
        dialog_len = len(a_dialog)
        for idx in range(dialog_len-1):
            query_val = a_dialog.iloc[idx]
            queryBag += [str(query_val)]
            cum_queryBag += [queryBag[:idx+1]]

            response_val = a_dialog.iloc[idx+1]
            responseBag += [str(response_val)]

        # response 및 모든 대화 누적응답 기록 (all_respnoseBag) 추출
        all_responseBag += responseBag
        historyBag += cum_queryBag

        # historyBag 기준으로 full_dataset을 인덱싱하기 위해 인덱스 벡터 추출
        all_idx = list(a_dialog[:idx+1].index)
        indexBag += all_idx
        
        # full_dataset에 달려있는 act / emotion 컬럼은 query 기준이므로, response에 대한 act와 emotion 벡터 별도로 추출
        response_idx = list(a_dialog[1:idx+2].index)
        all_act_r_Bag += list(full_dataset['act_q'].iloc[response_idx])
        all_emo_r_Bag += list(full_dataset['emo_q'].iloc[response_idx])
   
    return pd.DataFrame({'dialog_history' : historyBag}), indexBag, all_responseBag, pd.DataFrame({'act_r' : all_act_r_Bag}), pd.DataFrame({'emo_r' : all_emo_r_Bag})


'''
첫 문장 추출 함수
'''
import re
def extract_first_sentence(paragraph):
    # Use regular expressions to split the paragraph into sentences
    # sentences = re.split(r'(?<=[.!?;:()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    sentences = re.split(r'(?<=[.!?()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    
    # Extract the first sentence (remove leading/trailing whitespace)
    first_sentence = sentences[0].strip()
    
    return first_sentence

def extract_n_sentences(paragraph, n):
    # Use regular expressions to split the paragraph into sentences
    # sentences = re.split(r'(?<=[.!?;:()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    sentences = re.split(r'(?<=[.!?()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    
    # Extract the first n sentences (remove leading/trailing whitespace)
    extracted_sentences = [sentence.strip() for sentence in sentences[:n]]

    # Get full sentence
    full_sentence = ""
    for extract_sentence in extracted_sentences:
        full_sentence += str(extract_sentence)

    return full_sentence

def get_first_n_words(sentence, n):
    words = sentence.split()
    if n <= len(words):
        return ' '.join(words[:n])
    else:
        return sentence

# 데이터를 quantile 또는 trunc_len에 따라 절삭하는 함수
def truncate_data(pd_input, quantile: float):
    
    # seq_len_list = pd_input.apply(lambda x : len(x))
    seq_len_list = pd_input.apply(lambda x : len(x.split(' ')))
    TruncLen = seq_len_list.quantile(q=quantile)
    target_idx = np.where(seq_len_list <= TruncLen)[0]
    trunc_pd_input = pd_input.iloc[target_idx]

    return trunc_pd_input, TruncLen, target_idx

def get_token(token_dict, target_word):
    # print('target_word :', target_word)
    token = np.where(np.array(list(token_dict.values())) == target_word)[0][0]
    return token

def del_no_first_bos_row(data, token_dict):
    all_row_idx = np.arange(data.shape[0])
    target_idx = np.where(data[:, 0] != get_token(token_dict, '[bos]'))[0]
    if len(target_idx) > 0:
        del_idx = target_idx[0]
        del_data = data[all_row_idx != del_idx]        
    else:
        del_data = data

    return del_data

def add_token_list_in_dict(add_word_list, token_dict):
    '''
    원하는 단어 리스트를 토큰 사전에 추가.
    [pad]는 0번에, 그 외 단어들은 사전의 뒷쪽에 추가하는 함수
    '''
    pad_item = {0:'[pad]'}
    pad_item.update(token_dict)
    token_dict = pad_item
    last_idx = len(token_dict)-1

    add_token_dict_rev = { word:idx+last_idx for idx, word in enumerate(add_word_list) if word != '[pad]'}
    add_token_dict = dict(map(reversed, add_token_dict_rev.items()))

    token_dict.update(add_token_dict)

    return token_dict

def indexing_eos_token(inputs):
    '''
    [pad] 토큰이 0번으로 지정되어 있기 때문에, 시퀀스의 각 토큰을 time-step에 따라 누적합한 행렬 A를 만든 뒤,
    행렬 A에서 최대값이 등장하는 최초의 col_idx를 찾으면, col_idx + 1에 해당하는 지점이 각 시퀀스에서 [pad] 토큰이 최초로 등장한 지점.
    다시 말해, col_idx는 각 시퀀스에서 [pad] 토큰 바로 이전 토큰인 [eos] 토큰이 등장한 지점 (= [eos] 토큰의 col_idx).
    즉, 본 함수는 각 시퀀스에서 [eos] 토큰을 인덱싱를 뽑는 함수.
    '''
    A = tf.math.cumsum(inputs, axis = -1)
    eos_idx_vec = tf.argmax(A, axis = -1)
    return tf.cast(eos_idx_vec, dtype = tf.int32)

def combine_pad_n_subseq_mask(pad_mask, subseq_mask):
    '''
    pad_mask와 subseq_mask를 합치는 함수
    pad_mask : (batch_size, 1, 1, seq_len)
    subseq_mask : (seq_len, seq_len)
    '''

    batch_size = pad_mask.shape[0]
    subseq_mask2 = subseq_mask[tf.newaxis, tf.newaxis, :, :]
    subseq_mask3 = tf.tile(subseq_mask2, multiples = [batch_size, 1, 1, 1])
    pad_n_subseq_mask = subseq_mask3 * pad_mask

    return pad_n_subseq_mask

'''
마스크 언어모델 훈련을 위한 마스킹 함수
'''
def get_masking_param(inputs):
    '''
    전체 데이터 셋 평균 토큰 길이를 토대로 마스킹 파라미터인 평균 마스크 갯수 정의
    '''

    nonzero_entry_indices = tf.cast(tf.where(tf.logical_not(inputs == 0)), dtype = tf.int32)
    one_vector = tf.ones(shape = tf.shape(nonzero_entry_indices)[0])
    onehot_matrix = tf.scatter_nd(indices = nonzero_entry_indices, updates = one_vector, shape = tf.shape(inputs))
    num_of_nonzero_entry = tf.reduce_sum(onehot_matrix, axis = -1) - 2                  # 2를 빼주는 것은 [bos]와 [eos]를 제외해주기 위함
    mean_num_mask = int(tf.reduce_mean(num_of_nonzero_entry) * 0.15)                    # X 0.15 하는 이유는, BERT 논문을 따름.

    return mean_num_mask

def mask_level_seletor(thredshold = 0.5):
    u = tf.random.uniform(shape = [1], minval=0, maxval=1)
    if u >= thredshold:
        mask_lev = 'token'
    else:
        mask_lev = 'span'
    return mask_lev

def poisson_mask_generator(inputs, lambda_, token_dict, masking_level = 'token'):

    '''
    1) masking_level == 'token'의 경우
    mean_num_mask를 lambda로 하는 poisson 분포로부터, input의 각 샘플 시퀀스에서 마스킹 될 "토큰의 갯수"를 샘플링한다.
    이 후, 각 시퀀스 (= row_idx) 별로 샘플링 된 마스크의 갯수만큼 랜덤하게 col_idx를 샘플링 한다. (= 즉, row_idx에 매칭되는 col_idx의 갯수가 다름)
    이렇게 구축된 < row_idx X col_idx > 정보를 가지고 inputs[row_idx, col_idx] = 0 을 덧씌워준다. (= 0 토큰은 [pad] 토큰이므로.)

    2) masking_level == 'span'의 경우
    mean_num_mask를 lambda로 하는 poisson 분포로부터, input의 각 샘플 시퀀스에서 마스킹 될 "토큰의 갯수"를 샘플링한다.
    이 후, 각 시퀀스 (= row_idx) 별로 샘플링 된 마스크의 갯수를 고려하여 mask_span 의 시작지점으로서 활용가능한 col_idx의 상한 (= upper_limit_idx) 및 하한 (= lower_limit_idx)을 결정한다.
    결정된 col_idx의 상한과 하한을 활용하여 mask_span을 샘플링 한다.
    이렇게 구축된 < row_idx X col_idx > 정보를 가지고 inputs[row_idx, col_idx] = 0 을 덧씌워준다. (= 0 토큰은 [pad] 토큰이므로.)
    '''
    if masking_level == 'token':
        print('Let\'s do token-masking !')

        batch_size = tf.shape(inputs)[0]

        # 마스킹 적용할 토큰들의 row_idx 뽑기
        num_mask_vec = tf.cast(tf.random.poisson(shape = [batch_size], lam = [lambda_]), dtype = tf.int32)    # 각 시퀀스 별 mask 갯수 (= num_mask)들의 벡터를 포아송 분포로부터 샘플링
        batch_row_idx = tf.repeat(tf.range(batch_size), repeats = tf.squeeze(num_mask_vec))                   # 각 시퀀스 별 num_mask 만큼 해당 시퀀스의 row_idx를 복제

        # 마스킹 적용할 "토큰들"의 col_idx 뽑기
        eos_idx_vec = indexing_eos_token(inputs)                                            # 각 시퀀스 별 [eos] 토큰 인덱스들의 벡터
        repeated_eos_idx_vec = tf.repeat(eos_idx_vec, repeats = tf.squeeze(num_mask_vec))   # 각 시퀀스 별 [eos] 토큰 인덱스를 각 시퀀스 별 num_mask 만큼 복제
        post_bos_idx_vec = tf.repeat([1], repeats = len(repeated_eos_idx_vec))              # 각 시퀀스 별 [bos] 토큰 인덱스 (= 1번 컬럼인덱스)를 각 시퀀스 별 num_mask 만큼 복제
        batch_col_idx = np.random.randint(size = len(repeated_eos_idx_vec), low = post_bos_idx_vec, high = repeated_eos_idx_vec)    # 각 시퀀스 별 num_mask 만큼의 랜덤 int를 col_idx로써 부여

        # 마스킹 적용할 "토큰들"의 row_idx X col_idx 배열 만들기
        batch_mask_idx = tf.concat([batch_row_idx[:, tf.newaxis], batch_col_idx[:, tf.newaxis]], axis = -1)  # 배치 내 각 시퀀스 별로 [mask] 토큰이 들어갈 row_col_idx.

        # 마스킹 벡터를 만들어, inputs의 row_idx X col_idx 에 해당하는 지점에 덧씌워주기
        # mask_vector = tf.ones(shape = tf.shape(batch_mask_idx)[0]) * 9576   # (9576이 '[mask]' 토큰일 경우; get_token(token_dict, '[mask]') 함수로 확인가능)
        mask_vector = tf.ones(shape = tf.shape(batch_mask_idx)[0]) * get_token(token_dict, '[mask]')   # (9576이 '[mask]' 토큰일 경우; get_token(token_dict, '[mask]') 함수로 확인가능)
        mask_vector = tf.cast(mask_vector, dtype = tf.int32)
        masked_inputs = tf.tensor_scatter_nd_update(inputs, indices = batch_mask_idx, updates = mask_vector)

    elif masking_level == 'span':
        print('Let\'s do span-masking !')

        batch_size = tf.shape(inputs)[0]

        # 마스킹 적용할 토큰들의 row_idx 뽑기
        num_mask_vec = tf.cast(tf.random.poisson(shape = [batch_size], lam = [lambda_]), dtype = tf.int32)    # 각 시퀀스 별 mask 갯수 (= num_mask)들의 벡터를 포아송 분포로부터 샘플링
        batch_row_idx = tf.repeat(tf.range(batch_size), repeats = tf.squeeze(num_mask_vec))                         # 각 시퀀스 별 num_mask 만큼 해당 시퀀스의 row_idx를 복제

        # [pad] 제외 토큰들 중 마지막 토큰 (= [eos])의 인덱스 뽑기
        eos_idx_vec = indexing_eos_token(inputs)

        # # num_mask_vec 값 조정
        # num_mask_vec = tf.squeeze(num_mask_vec).numpy()
        # eos_idx_vec = eos_idx_vec.numpy()
        # num_mask_vec[eos_idx_vec - num_mask_vec <= 1] = int(2)
        # num_mask_vec = num_mask_vec[:, tf.newaxis]

        # 마스킹 span 시작지점으로 활용가능한 col_idx의 상한값 결정
        upper_limit_of_sampling_idx = eos_idx_vec - tf.squeeze(num_mask_vec)
        upper_limit_idx = upper_limit_of_sampling_idx.numpy()

        # upper_limit 값 조정
        upper_limit_idx[upper_limit_idx <= 1] = int(2)      # -> 원칙적으로, num_mask_vec + 1의 원소값이 eos_idx_vec의 원소값 보다 항상 작아야 함
                                                            # -> 즉, eos_idx_vec - num_mask_vec <= 1 인 경우는 이 원칙에 위배됨
                                                            # -> 한편, upper_limit_idx = eos_idx_vec - num_mask_vec.
                                                            # -> 따라서, upper_limit[upper_limit_idx <= 1] = int(2) 로 할당 해주어야 함
                                                            # --> 2를 할당하는 이유는 아래 np.random.randint() 함수와 관련이 있는데, 
                                                            # --> low = [1, 1, ..., 1] 이 고정이고, random sampling이 가능하려면 high = [최소 2이상, 최소 2이상, ..., 최소 2이상] 이어야 하기 때문임.
                                                            # --> 이렇게 2로 고정함에 따라 해당 시퀀스는 [bos]와 [eos] 사이의 모든 토큰들을 마스킹하게 됨.


        # 마스킹 span 시작지점으로 활용가능한 col_idx의 하한값 결정
        lower_limit_idx = tf.repeat(1, repeats = len(upper_limit_idx)).numpy()

        # col_idx의 하한과 상한 사이에서 임의의 마스킹 시작점 샘플링
        mask_begin_idx = np.random.randint(size = len(lower_limit_idx), low = lower_limit_idx, high = upper_limit_idx)
        mask_end_idx = mask_begin_idx + tf.squeeze(num_mask_vec).numpy()

        # 마스킹 적용할 "스팬들"의 col_idx 샘플링
        mask_span_list = tf.ragged.range(mask_begin_idx, mask_end_idx).to_list()
        mask_span_list_flatten = np.concatenate(mask_span_list)
        batch_col_idx = tf.cast(mask_span_list_flatten, dtype = tf.int32)

        # 마스킹 적용할 "스팬들"의 row_idx X col_idx 배열 만들기
        batch_mask_idx = tf.concat([batch_row_idx[:, tf.newaxis], batch_col_idx[:, tf.newaxis]], axis = -1)  # 배치 내 각 시퀀스 별로 [mask] 토큰이 들어갈 row_col_idx.

        # 마스킹 벡터를 만들어, inputs의 row_idx X col_idx 에 해당하는 지점에 덧씌워주기
        # mask_vector = tf.ones(shape = tf.shape(batch_mask_idx)[0]) * 9576   # (9576 이 '[mask]' 토큰일 경우; get_token(token_dict, '[mask]') 함수로 확인가능)
        mask_vector = tf.ones(shape = tf.shape(batch_mask_idx)[0]) * get_token(token_dict, '[mask]')   # (9576 이 '[mask]' 토큰일 경우; get_token(token_dict, '[mask]') 함수로 확인가능)
        mask_vector = tf.cast(mask_vector, dtype = tf.int32)
        masked_inputs = tf.tensor_scatter_nd_update(inputs, indices = batch_mask_idx, updates = mask_vector)

    return masked_inputs, batch_mask_idx

'''
보상모델 훈련을 위한 가짜 시퀀스 생성함수
'''
def create_fake_dataset(inputs, fake_gen_type = 'all-or-not'):

    '''
    inputs의 절반만을 샘플링
    '''
    # 실제 시퀀스 생성
    real_inputs = copy.deepcopy(inputs)
    # random_shuffle_idx = np.random.choice(a = np.arange(real_inputs.shape[0]), size = real_inputs.shape[0] // 2, replace = False)
    # half_real_inputs = real_inputs[random_shuffle_idx, :]

    # 가짜 시퀀스 생성 (: epoch마다 span-shuffle or token-delete 둘 중 하나만 수행)
    if fake_gen_type == 'all-or-not':
        fake_type = fake_type_seletor(thredshold = 0.5)
        fake_inputs = fake_generator(real_inputs, fake_type = fake_type)
    
    # 가짜 시퀀스 생성 (: epoch마다 span-shuffle과 token-delete 반반 섞어 진행)
    elif fake_gen_type == 'half-and-half':
        fake_inputs = fake_generator(real_inputs, fake_type = None)

    # 실제/가짜 라벨 생성
    real_labels = np.repeat(1, real_inputs.shape[0])                        # True_label = 1
    fake_labels = np.repeat(0, fake_inputs.shape[0])                        # Fake_label = 0

    return real_inputs, real_labels, fake_inputs, fake_labels

def fake_type_seletor(thredshold = 0.5):
    u = tf.random.uniform(shape = [1], minval=0, maxval=1)
    # u = 0.51
    if u >= thredshold:
        fake_type = 'shuffle'
    else:
        fake_type = 'delete'
    return fake_type

def fake_generator(inputs, fake_type = 'shuffle'):

    if fake_type != None:
        '''
        fake 시퀀스를 매 epoch마다 span-shuffle 또는 token-delete 둘 중 하나만 수행
        '''
        print('\n')
        print('fake_gen_type : all-or-not')

        if fake_type == 'shuffle':
            print('Let\'s do span-shuffling !')

            # fake_inputs = span_shuffling(inputs)
            fake_inputs = span_shuffling_ver2(inputs)

        elif fake_type == 'delete':
            print('Let\'s do token-deleting !')

            fake_inputs = token_deleting(inputs)

    else:
        '''
        fake 시퀀스를 매 epoch마다 span-shuffle과 token-delete 반반 섞어 수행
        '''
        print('\n')
        print('fake_gen_type : half-and-half')

        thredshold = inputs.shape[0] // 2
        random_shuffle_idx = np.random.choice(a = np.arange(inputs.shape[0]), size = inputs.shape[0], replace = False)

        target_idx1 = random_shuffle_idx[:thredshold]
        target_idx2 = random_shuffle_idx[thredshold:]

        fake_inputs1 = span_shuffling_ver2(inputs[target_idx1, :])
        fake_inputs2 = token_deleting(inputs[target_idx2, :])
        fake_inputs = tf.concat([fake_inputs1, fake_inputs2], axis = 0)

    return fake_inputs

# def span_shuffling(inputs):

#     '''
#     스팬의 컬럼 인덱스 구축하기
#     '''
#     eos_idx = indexing_eos_token(inputs)                                                                # 각 시퀀스 별로 eos 토큰의 인덱스
#     ones_vec = tf.ones(shape = tf.shape(eos_idx), dtype = tf.int32)                                     # 원 벡터 얻기
#     span_begin_idx = np.random.randint(low = ones_vec, high = eos_idx - 1, size = len(eos_idx))         # span 시작점 (= 시작점이 가능한 최대 인덱스는 eos_idx-2 에 해당하도록 세팅. 이렇게 해야 스팬의 길이 최소 2로 유지됨)
#     span_idx = tf.ragged.range(span_begin_idx, eos_idx)                                                 # span 인덱스 정의
#     col_idx = np.concatenate(span_idx.to_list())                                                        # span 인덱스 = col_idx

#     '''
#     스팬의 로우 인덱스 (= 각 행별 스팬 길이 만큼의 오름차순 인덱스) 구축하기
#     '''
#     len_vec = [ len(i) for i in span_idx.to_list() ]
#     batch_size = inputs.shape[0]
#     row_idx = tf.repeat(tf.range(batch_size), repeats = len_vec)

#     '''
#     스팬의 배치 인덱스
#     '''
#     batch_idx = tf.concat([row_idx[:, tf.newaxis], col_idx[:, tf.newaxis]], axis = -1)

#     '''
#     스팬에 해당하는 값들을 셔플링 및 셔플링 된 토큰을 할당
#     '''
#     # batch_idx를 랜덤하게 섞기 (= row_idx는 고정이고 col_idx만 섞여야 함)
#     shuffled_col_idx = [np.random.choice(vec, size = len(vec), replace=False) for idx, vec in enumerate(span_idx.to_list())]
#     shuffled_col_idx = np.concatenate(shuffled_col_idx)
#     shuffled_batch_idx = tf.concat([row_idx[:, tf.newaxis], shuffled_col_idx[:, tf.newaxis]], axis = -1)
    
#     # shuffled_batch_idx를 따라 토큰 값들을 gather_nd 하기
#     shuffled_tokens = tf.gather_nd(inputs, shuffled_batch_idx)

#     # gather_nd 한 값들을 원래 batch_idx에 tensor_scatter_nd_updates 해주기 (= 셔플된 토큰을 기존의 인덱스에 할당)
#     span_shuffled_inputs = tf.tensor_scatter_nd_update(tensor = inputs, indices = batch_idx, updates = shuffled_tokens)

#     return span_shuffled_inputs

def span_shuffling_ver2(inputs):

    '''
    스팬의 컬럼 인덱스 구축하기
    '''
    eos_idx = indexing_eos_token(inputs)                                                                            # 각 시퀀스 별로 eos 토큰의 인덱스
    ones_vec = tf.ones(shape = tf.shape(eos_idx), dtype = tf.int32)                                                 # 원 벡터 얻기
    span_begin_idx = np.random.randint(low = ones_vec, high = eos_idx - 1, size = len(eos_idx))                     # span 시작점 (= 시작점이 가능한 최대 인덱스는 eos_idx-2 에 해당하도록 세팅.)
                                                                                                                    # (이렇게 해야 span의 길이가 최소 2로 유지됨.)

    span_end_idx = np.random.randint(low = span_begin_idx + 1, high = eos_idx, size = len(eos_idx))                 # span 종료점 (= 종료점이 가능한 최소 인덱스는 시작점 +1, 최대 인덱스는 eos_idx-1 에 해당하도록 세팅.)
                                                                                                                    # (시작점 +1 : span의 시작점을 제외함으로써 span의 종료점과 시작점이 겹치는 경우를 방지할 수 있음.
                                                                                                                    # (eos_idx -1 : span의 종료점이 발생가능한 인덱스를 eos_idx 바로 이전 인덱스까지 허용함으로써 span의 최대범위를 문장 전체의 길이로 허용.)

    span_idx = tf.ragged.range(span_begin_idx, span_end_idx + 1)                                                    # span 인덱스 정의 (= span_end_idx + 1 을 함으로써 span_end_idx 까지 span_idx 안에 포함시켜줌)
    col_idx = np.concatenate(span_idx.to_list())                                                                    # span 인덱스 = col_idx

    '''
    스팬의 로우 인덱스 (= 각 행별 스팬 길이 만큼의 오름차순 인덱스) 구축하기
    '''
    len_vec = [ len(i) for i in span_idx.to_list() ]
    batch_size = inputs.shape[0]
    row_idx = tf.repeat(tf.range(batch_size), repeats = len_vec)


    '''
    스팬의 배치 인덱스
    '''
    batch_idx = tf.concat([row_idx[:, tf.newaxis], col_idx[:, tf.newaxis]], axis = -1)


    '''
    스팬에 해당하는 값들을 셔플링 및 셔플링 된 토큰을 할당
    '''
    # batch_idx를 랜덤하게 섞기 (= row_idx는 고정이고 col_idx만 섞여야 함)
    # shuffled_col_idx = [np.random.choice(vec, size = len(vec), replace=False) for idx, vec in enumerate(span_idx.to_list())]
    # shuffled_col_idx = [np.random.permutation(vec) for idx, vec in enumerate(span_idx.to_list())]
    shuffled_col_idx = []
    for vec in span_idx.to_list():
        done = True
        while(done == True):
            perm_vec = np.random.permutation(vec)
            if np.array_equal(vec, perm_vec) == False:
                done = False

        shuffled_col_idx.append(perm_vec)

    shuffled_col_idx = np.concatenate(shuffled_col_idx)
    shuffled_batch_idx = tf.concat([row_idx[:, tf.newaxis], shuffled_col_idx[:, tf.newaxis]], axis = -1)

    # shuffled_batch_idx를 따라 토큰 값들을 gather_nd 하기
    shuffled_tokens = tf.gather_nd(inputs, shuffled_batch_idx)

    # gather_nd 한 값들을 원래 batch_idx에 tensor_scatter_nd_updates 해주기 (= 셔플된 토큰을 기존의 인덱스에 할당)
    span_shuffled_inputs = tf.tensor_scatter_nd_update(tensor = inputs, indices = batch_idx, updates = shuffled_tokens)

    return span_shuffled_inputs

def token_deleting(inputs):

    '''
    각 시퀀스 별로 삭제할 토큰의 갯수를 설정하기 
    '''
    # 각 시퀀스의 [1, eos_idx) 길이의 절반에 해당하는 정수값을 삭제할 토큰 갯수의 상한으로 설정
    eos_idx = indexing_eos_token(inputs)
    inputs_seq_len = eos_idx - 1
    inputs_seq_median_len = inputs_seq_len // 2

    # 1을 삭제할 토큰 갯수의 하한으로 설정 (= 무조건 하나 이상의 토큰 삭제)
    ones_vec = tf.ones(shape = tf.shape(inputs_seq_median_len), dtype = tf.int32)
    new_ones_vec = ones_vec.numpy()

    '''
    예외 경우 (= exceptional case) 처리
    --> 토큰 갯수의 상한이 1보다 작거나 같은 경우 (하한도 1임으로) 샘플링 범위가 [1, 1)로 설정됨
    --> 위와 같이 설정된 범위에서는 샘플링이 불가능
    # --> 따라서, 상한이 1보다 작거나 같은 경우, 하한을 0 / 상한을 1로 고정하여 샘플링 범위를 [0, 1)로 설정
    # --> 이 경우 무조건 0이 샘플링 되어 삭제할 토큰의 갯수는 0이 됨.
    --> 따라서, 상한이 1보다 작거나 같은 경우, 하한을 1 / 상한을 2로 고정하여 샘플링 범위를 [1, 2)로 설정
    --> 이 경우 무조건 1이 샘플링 되어 삭제할 토큰의 갯수는 1이 됨. 즉, 무조건 하나 이상의 토큰을 삭제하게 됨.
    '''
    # 하한을 0으로 고정 
    expt_idx = tf.squeeze(tf.where(inputs_seq_median_len <= 1)).numpy()
    # new_ones_vec[expt_idx] = 0
    new_ones_vec[expt_idx] = 1

    # 상한을 1으로 고정
    new_inputs_seq_median_len = inputs_seq_median_len.numpy()
    # new_inputs_seq_median_len[expt_idx] = 1
    new_inputs_seq_median_len[expt_idx] = 2

    # 삭제할 토큰의 갯수 샘플링
    num_delete = np.random.randint(low = new_ones_vec, high = new_inputs_seq_median_len, size = len(new_inputs_seq_median_len))

    # # 삭제하지 말아야 할 (= 살려야 할) 토큰의 갯수 샘플링
    # num_not_delete = inputs_seq_len - num_delete

    '''
    각 시퀀스에서 [bos], [eos]를 제외한 토큰들 중 삭제할 토큰의 갯수만큼 임의의 토큰 삭제
    즉, 각 시퀀스에서 [bos], [eos]를 제외한 토큰들 중 살려야 할 토큰의 갯수만큼 임의의 토큰 선택
    '''
    # 삭제할 토큰의 col_idx로 구성된 batch_del_idx 뽑기
    # [bos]의 idx는 0임으로 미포함 = ones_vec
    # [eos]의 idx는 ceiling으로 샘플링에서 제외 = range(, eos_idx)
    seq_idx = tf.ragged.range(ones_vec, eos_idx)
    col_idx = np.concatenate([ np.random.choice(vec, size = num_delete[idx], replace=False) for idx, vec in enumerate(seq_idx.to_list()) ])
    row_idx = tf.repeat(tf.range(tf.shape(inputs)[0]), repeats = num_delete)
    batch_del_idx = tf.concat([row_idx[:, tf.newaxis], col_idx[:, tf.newaxis]], axis = -1)
    batch_del_idx = batch_del_idx.numpy()

    # 모든 토큰의 col_idx로 구성된 batch_all_idx 뽑기
    zeros_vec = tf.zeros(shape = tf.shape(inputs)[0], dtype = tf.int32)
    seq_idx = tf.ragged.range(zeros_vec, eos_idx + 1)                                           # eos_idx + 1 해주기

    col_idx = np.concatenate(seq_idx.to_list())
    # row_idx = tf.repeat(tf.range(tf.shape(inputs)[0]), repeats = inputs_seq_len)
    row_idx = tf.repeat(tf.range(tf.shape(inputs)[0]), repeats = inputs_seq_len + 2)            # inputs_seq_len + 2 해주기

    batch_all_idx = tf.concat([row_idx[:, tf.newaxis], col_idx[:, tf.newaxis]], axis = -1)
    batch_all_idx = batch_all_idx.numpy()

    # 살려야 할 토큰의 col_idx로 구성된 batch_not_del_idx 뽑기
    # batch_all_idx와 batch_del_idx의 차집합, 즉 batch_not_del_idx 뽑기
    # column들을 tuple 형태로 row-wise하게 묶은 뒤, 각 tuple을 하나의 원소로 보고 batch_all_idx에서 batch_del_idx와 겹친 원소들 제거
    tuple_del_idx = batch_del_idx.view([('', batch_del_idx.dtype)] * batch_del_idx.shape[1])
    tuple_all_idx = batch_all_idx.view([('', batch_all_idx.dtype)] * batch_all_idx.shape[1])
    batch_not_del_idx = np.setdiff1d(tuple_all_idx, tuple_del_idx).view(batch_all_idx.dtype).reshape(-1, batch_all_idx.shape[1])
    batch_not_del_idx = tf.cast(batch_not_del_idx, dtype = tf.int32)

    '''
    살려야 할 토큰들에 해당하는 값들로 배치 재구축;
    삭제된 만큼의 토큰은 뒤에 [pad]로 붙음
    '''
    # 삭제할 토큰 외 다른 모든 토큰들을 토대로 inputs를 재구축하기 
    not_del_tokens = tf.gather_nd(params = inputs, indices = batch_not_del_idx)
    reranged_batch_not_del_col_idx = rerange_col_idx(batch_not_del_idx)
    batch_not_del_row_idx = tf.cast(batch_not_del_idx[:, 0], dtype = tf.int32)
    reranged_batch_not_del_idx = tf.concat([batch_not_del_row_idx[:, tf.newaxis], reranged_batch_not_del_col_idx[:, tf.newaxis]], axis = -1)

    token_deleted_inputs = tf.scatter_nd(indices=reranged_batch_not_del_idx, updates = not_del_tokens, shape = tf.shape(inputs))

    return token_deleted_inputs

'''
CANVAS 생성 및 Levenshtein Operation 함수들
'''
@tf.function
def create_canvas(inputs):
    '''
    canvas 생성 함수
    inputs으로 액션 또는 시퀀스
    '''

    # 캔버스 형상 관련 정보 추출
    batch_size = tf.shape(inputs)[0]                        # 배치 사이즈
    canvas_idx, canvas_seq_len = get_canvas_idx(inputs)     # 캔버스 인덱스, 캔버스 시퀀스 길이
    canvas_shape = (batch_size, canvas_seq_len)             # 캔버스 쉐이프

    # 캔버스 생성
    canvas_with_zero = get_zero_buffer_canvas(inputs, canvas_idx, canvas_shape)

    return canvas_with_zero

@tf.function
def get_canvas_idx(inputs):
    '''
    원시적인 형태의 canvas (= 원핫 캔버스) 생성 및 이로부터 buffer가 들어갈 자리와 그 외 자리를 구분하는 인덱스 추출
    '''

    batch_size = tf.shape(inputs)[0]            # 배치 사이즈
    batch_seq_len = tf.shape(inputs)[1]         # 시퀀스 길이

    inputs_canvas = tf.ones(shape = (batch_size, 2 * batch_seq_len - 1))                    # 캔버스 (캔버스 시퀀스 길이 = 2 * 배치 시퀀스 길이 - 1); 1벡터
    canvas_seq_len = tf.shape(inputs_canvas)[1]                                             # 캔버스 시퀀스 길이
    
    # 캔버스 짝수 컬럼 인덱스 타일링
    canvas_col_idx = tf.range(canvas_seq_len)                                               # 캔버스 컬럼 인덱스 (= col_idx)
    canvas_odd_col_idx_onehot = canvas_col_idx % 2                                          # 캔버스 홀수 컬럼 인덱스 (= odd_col_idx) 원핫
    canvas_even_col_idx_onehot = 1 - canvas_odd_col_idx_onehot                              # 캔버스 짝수 컬럼 인덱스 (= even_col_idx) 원핫
    canvas_even_col_idx = tf.cast(tf.where(canvas_even_col_idx_onehot), dtype = tf.int32)   # 캔버스 짝수 컬럼 인덱스 (= 1, 3, 5, ...)
    col_even_idx_tile = tf.tile(canvas_even_col_idx, multiples = (batch_size, 1))           # 배치 사이즈 만큼 canvas_even_col_idx 복사

    # 캔버스 로우 인덱스 타일링
    canvas_even_col_idx_len = len(canvas_even_col_idx)
    canvas_row_idx = tf.range(batch_size)
    row_idx_tile = tf.repeat(canvas_row_idx, canvas_even_col_idx_len)[:, tf.newaxis]
    row_idx_tile = tf.cast(row_idx_tile, dtype = tf.int32)    

    # 캔버스 <로우 X 짝수 컬럼> 인덱스
    batch_canvas_idx = tf.concat([row_idx_tile, col_even_idx_tile], axis = -1)

    return batch_canvas_idx, canvas_seq_len

@tf.function
def get_zero_buffer_canvas(inputs, canvas_idx, canvas_shape):
    '''
    canvas 구축시, 홀수 컬럼 인덱스에 토큰 0 을 할당하는 함수 (= 제로 패딩하는 함수).
    -> canvas 내 짝수 컬럼에는 기존의 input 값들을 할당.
    -> canvas 내 홀수 컬럼은 buffer (= [mask] 토큰이 들어갈 후보지)로써 활용하는데, 이 buffer 자리에 일단 [pad] 토큰을 일단 할당 = zero_buffer
    '''

    # 제로패드 캔버스 구축 (= 캔버스의 홀수 컬럼 인덱스가 제로 패드 (= [pad])인 캔버스)
    flatten_inputs = tf.squeeze(tf.reshape(inputs, shape = (-1, 1)))
    canvas_with_zero = tf.scatter_nd(indices = canvas_idx, updates = flatten_inputs, shape = canvas_shape)
    canvas_with_zero = tf.cast(canvas_with_zero, dtype = tf.int32)

    return canvas_with_zero

@tf.function
def apply_lev_operation(seq_inputs, action_inputs, token_dict):
    '''
    Levenshtein 연산자를 적용하는 함수
    
    seq_inputs : 시퀀스 배치
    action_inputs : 시퀀스 배치의 각 토큰 대신 Levenshtein 연산자가 맵핑 된 배치
    '''
    action_inputs = insert_bos_n_eos_token(seq_inputs, action_inputs, token_dict)
    action_inputs = pad_after_eos(seq_inputs, action_inputs, token_dict)
    # print('action_inputs : ', action_inputs)

    # 캔버스 생성
    action_canvas_with_zero = create_canvas(action_inputs)
    # print('action_canvas_with_zero : ', action_canvas_with_zero)
    seq_canvas_with_zero = create_canvas(seq_inputs)

    '''
    1) INSERT 오퍼레이션 수행
    1-1) [INS_F]-1, [INS_B]+1, [INS_A]+-1 -> [REP]
    1-2) [INS_F], [INS_B], [INS_A] -> [KEP]
    '''
    action_canvas_inserted = INSERT_operation(action_canvas_with_zero, token_dict)

    '''
    2) [REP] -> [mask]
    '''
    action_canvas_masked = replace_REP_with_mask(action_canvas_inserted, token_dict)

    '''
    3) action_canvas_masked 기준 [KEP] 토큰 자리에 해당하는 seq_canvas_with_zero의 값들을 action_canvas_masked에 업데이트
    '''
    seq_canvas_masked = replace_KEP_with_token(seq_canvas_with_zero, action_canvas_masked, token_dict)

    '''
    4) [DEL], [pad] -> None (= 기존의 캔버스와 동일한 shape를 가지되, [DEL]에 의한 토큰 삭제가 반영된 캔버스 생성)
    '''
    seq_canvas_deleted = slice_del_n_pad(seq_canvas_masked, token_dict)

    return seq_canvas_deleted, action_canvas_inserted

@tf.function
def insert_bos_n_eos_token(seq_inputs, action_inputs, token_dict):
    # action_inputs = action_inputs.numpy()
    # action_inputs[:, 0] = get_token(token_dict, '[bos]') 
    bos_token = get_token(token_dict, '[bos]')
    bos_idx = tf.concat([tf.range(tf.shape(action_inputs)[0])[:, tf.newaxis], tf.zeros(shape = tf.shape(action_inputs)[0], dtype = tf.int32)[:, tf.newaxis]], axis = 1)
    updates_val = tf.cast(tf.repeat(bos_token, tf.shape(bos_idx)[0]), dtype = tf.int32)
    action_inputs = tf.tensor_scatter_nd_update(tensor = action_inputs, indices = bos_idx, updates = updates_val)

    eos_token = get_token(token_dict, '[eos]')
    eos_idx = tf.where(seq_inputs == eos_token)
    updates_val = tf.cast(tf.repeat(eos_token, tf.shape(eos_idx)[0]), dtype = tf.int32)
    action_inputs = tf.tensor_scatter_nd_update(tensor = action_inputs, indices = eos_idx, updates = updates_val)

    return action_inputs

@tf.function
def pad_after_eos(seq_inputs, action_inputs, token_dict):
    
    pad_token = get_token(token_dict, '[pad]')
    pad_idx = tf.where(seq_inputs == pad_token)
    update_vals = tf.cast(tf.repeat(pad_token, tf.shape(pad_idx)[0]), dtype = tf.int32)
    action_inputs = tf.tensor_scatter_nd_update(tensor = action_inputs, indices = pad_idx, updates = update_vals)

    return action_inputs

@tf.function
def INSERT_operation(canvas_with_zero, token_dict):
    '''
    [INS] 관련 오퍼레이션 함수 (e.g., [INS_F], [INS_B], [INS_A]).
    이 때, 본 함수에 입력되는 canvas_with_zero는 액션 벡터에 대해 제로버퍼가 적용된 캔버스임.
    '''

    '''
    1) [INS_F] 처리 - 캔버스 액션 시퀀스에서 [INS_F]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t-1에 해당하는 곳에 [REP] 넣어주기
    '''    
    # [INS_F]에 해당하는 토큰 값 및 인덱스 구하기
    ins_f_token = get_token(token_dict, '[INS_F]')                                              # [INS_F]에 해당하는 토큰
    ins_f_idx = tf.cast(tf.where(canvas_with_zero == ins_f_token), dtype = tf.int32)            # [INS_F]에 해당하는 토큰의 인덱스

    # [INS_F]의 한 시점 앞에 추가로 [REP]가 삽입될 인덱스 구하기
    # add_rep_idx = ins_f_idx.numpy()                             
    # add_rep_idx[:, 1] = add_rep_idx[:, 1] + 1                                                   
    target_idx = tf.concat([ tf.range(tf.shape(ins_f_idx)[0])[:, tf.newaxis], tf.ones(shape = tf.shape(ins_f_idx)[0], dtype = tf.int32)[:, tf.newaxis] ], axis = -1)
    update_vals = tf.ones(shape = tf.shape(ins_f_idx)[0], dtype = tf.int32)
    add_rep_idx = tf.tensor_scatter_nd_sub(tensor = ins_f_idx, indices = target_idx, updates = update_vals)

    # 인덱스에 [REP] 삽입하기
    rep_token = tf.repeat([get_token(token_dict, '[REP]')], repeats = tf.shape(add_rep_idx)[0])                                         # [REP]에 해당하는 토큰 값을 add_rep_idx의 row 갯수 만큼 복사
    canvas_with_ins_f = tf.tensor_scatter_nd_update(tensor = canvas_with_zero, indices = add_rep_idx, updates = rep_token)              # canvas_with_zero 의 add_rep_idx 위치에 복사된 [REP] 토큰 값을 할당


    '''
    2) [INS_B] 처리 - 캔버스 액션 시퀀스에서 [INS_B]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t+1에 해당하는 곳에 [REP] 넣어주기
    '''    
    # [INS_B]에 해당하는 토큰 값 및 인덱스 구하기
    ins_b_token = get_token(token_dict, '[INS_B]')                                              # [INS_B]에 해당하는 토큰
    ins_b_idx = tf.cast(tf.where(canvas_with_ins_f == ins_b_token), dtype = tf.int32)           # [INS_B]에 해당하는 토큰의 인덱스

    # [INS_B]의 한 시점 앞에 추가로 [REP]가 삽입될 인덱스 구하기
    # add_rep_idx = ins_b_idx.numpy()                             
    # add_rep_idx[:, 1] = add_rep_idx[:, 1] + 1                                                   
    target_idx = tf.concat([ tf.range(tf.shape(ins_b_idx)[0])[:, tf.newaxis], tf.ones(shape = tf.shape(ins_b_idx)[0], dtype = tf.int32)[:, tf.newaxis] ], axis = -1)
    update_vals = tf.ones(shape = tf.shape(ins_b_idx)[0], dtype = tf.int32)
    add_rep_idx = tf.tensor_scatter_nd_add(tensor = ins_b_idx, indices = target_idx, updates = update_vals)

    # 인덱스에 [REP] 삽입하기
    rep_token = tf.repeat([get_token(token_dict, '[REP]')], repeats = tf.shape(add_rep_idx)[0])                                         # [REP]에 해당하는 토큰 값을 add_rep_idx의 row 갯수 만큼 복사
    canvas_with_ins_f_b = tf.tensor_scatter_nd_update(tensor = canvas_with_ins_f, indices = add_rep_idx, updates = rep_token)           # canvas_with_ins_f 의 add_rep_idx 위치에 복사된 [REP] 토큰 값을 할당


    '''
    3) [INS_A] 처리 - 캔버스 액션 시퀀스에서 [INS_A]에 해당하는 t를 찾아, 캔버스 토큰 시퀀스에서 t-1, t+1에 해당하는 곳에 [REP] 넣어주기
    '''    
    # [INS_A]에 해당하는 토큰 값 및 인덱스 구하기
    ins_a_token = get_token(token_dict, '[INS_A]')                                              # [INS_A]에 해당하는 토큰
    ins_a_idx = tf.cast(tf.where(canvas_with_ins_f_b == ins_a_token), dtype = tf.int32)         # [INS_A]에 해당하는 토큰의 인덱스

    # [INS_A]의 한 시점 앞/뒤에 추가로 [REP]가 삽입될 인덱스 구하기
    # add_rep_idx = ins_a_idx.numpy()                                 
    # add_rep_idx = tf.repeat(add_rep_idx, repeats = 2, axis = 0).numpy()                                                                 # [INS_A]는 앞/뒤로 동시에 처리해주어야 하니까 add_rep_idx를 2번 복사
    # neg_pos_vector = tf.squeeze(tf.tile(np.array([[-1, 1]]), multiples= (1, ins_a_idx.shape[0]))).numpy()                               # neg_pos_vector는 각각 [INS_A]의 앞 (-1)과 뒤 (+1) 를 인덱싱 해주는 값
    # add_rep_idx[:, 1] = add_rep_idx[:, 1] + neg_pos_vector
    ins_a_idx2 = tf.cast(tf.repeat(ins_a_idx, repeats = 2, axis = 0), dtype = tf.int32)                                                                                    # [INS_A]는 앞/뒤로 동시에 처리해주어야 하니까 add_rep_idx를 2번 복사
    target_idx = tf.concat([ tf.range(tf.shape(ins_a_idx2)[0])[:, tf.newaxis], tf.ones(shape = tf.shape(ins_a_idx2)[0], dtype = tf.int32)[:, tf.newaxis] ], axis = -1)
    neg_pos_vector = tf.squeeze(tf.tile(tf.constant([[-1, 1]], dtype = tf.int32), multiples= (1, tf.shape(ins_a_idx)[0])))                                                                    # neg_pos_vector는 각각 [INS_A]의 앞 (-1)과 뒤 (+1) 를 인덱싱 해주는 값
    add_rep_idx = tf.tensor_scatter_nd_add(tensor = ins_a_idx2, indices = target_idx, updates = neg_pos_vector)

    # 인덱스에 [REP] 삽입하기
    rep_token = tf.repeat([get_token(token_dict, '[REP]')], repeats = tf.shape(add_rep_idx)[0])                                         # [REP]에 해당하는 토큰 값을 add_rep_idx의 row 갯수 만큼 복사
    canvas_with_ins_f_b_a = tf.tensor_scatter_nd_update(tensor = canvas_with_ins_f_b, indices = add_rep_idx, updates = rep_token)       # canvas_with_ins_f_b 의 add_rep_idx 위치에 복사된 [REP] 토큰 값을 할당

    '''
    4) [INS_F], [INS_B], [INS_A] --> [KEP] 처리
    '''    
    # [INS_F], [INS_B], [INS_A]에 해당하는 인덱스 구하기
    all_ins_idx = tf.where( tf.math.logical_or(tf.math.logical_or(canvas_with_zero == ins_f_token, canvas_with_zero == ins_b_token), canvas_with_zero == ins_a_token))

    # 인덱스에 [KEP] 삽입하기
    kep_token = tf.repeat([get_token(token_dict, '[KEP]')], repeats = tf.shape(all_ins_idx)[0])                                         # [KEP]에 해당하는 토큰 값을 add_rep_idx의 row 갯수 만큼 복사
    canvas_with_ins = tf.tensor_scatter_nd_update(tensor = canvas_with_ins_f_b_a, indices = all_ins_idx, updates = kep_token)           # canvas_with_ins_f_b_a 의 all_ins_idx 위치에 복사된 [KEP] 토큰 값을 할당

    return canvas_with_ins
    # return canvas_with_ins_f_b_a

@tf.function
def replace_REP_with_mask(canvas_inserted, token_dict):
    '''
    모든 [REP] 토큰들의 인덱스에 [mask] 토큰 값 (= 9576) 삽입하기
    '''

    rep_idx = tf.cast( tf.where( canvas_inserted == get_token(token_dict, '[REP]') ), dtype = tf.int32)                 # [REP] 토큰 인덱스 얻기

    flatten_mask_vector = tf.ones(tf.shape(rep_idx)[0], dtype = tf.int32) * get_token(token_dict, '[mask]')                 # [mask] 토큰을 [REP] 토큰 갯수만큼 복사

    canvas_masked = tf.tensor_scatter_nd_update(canvas_inserted, indices = rep_idx, updates = flatten_mask_vector)      # [REP] 토큰 인덱스 위치에 [mask] 토큰 삽입

    return canvas_masked

@tf.function
def replace_KEP_with_token(seq_canvas_with_zero, action_canvas_masked, token_dict):
    kep_token = get_token(token_dict, '[KEP]')
    kep_idx = tf.where(action_canvas_masked == kep_token)

    target_val = tf.gather_nd(params = seq_canvas_with_zero, indices = kep_idx)
    seq_canvas_masked = tf.tensor_scatter_nd_update(tensor = action_canvas_masked, indices = kep_idx, updates = target_val)

    return seq_canvas_masked

@tf.function
def slice_del_n_pad(canvas_masked, token_dict):

    '''
    1) 캔버스 액션 시퀀스에서 [DEL]과 [PAD]를 제외한 모든 다른 액션 (= [REP], [KEP], [bos], [eos])들의 batch_idx를 뽑기
    '''
    del_token = get_token(token_dict, '[DEL]')
    pad_token = get_token(token_dict, '[pad]')
    eos_token = get_token(token_dict, '[eos]')
    
    # '[DEL]' 와 '[pad]' 토큰을 제외한 모든 토큰들의 인덱스 및 그 값들
    idx_except_for_del_n_pad = tf.where( tf.logical_not( tf.math.logical_or( canvas_masked == del_token, canvas_masked ==  pad_token ) ) )
    val_except_for_del_n_pad = tf.gather_nd( params = canvas_masked, indices = idx_except_for_del_n_pad )

    '''
    2) batch_idx의 col_idx를 각 row_idx별로 0부터 오름차순으로 범위 재지정하기
    '''
    # idx_except_for_del_n_pad 값에서, col_idx 부분에 끊긴 부분들을 이어주기 (= 0부터 시작해서 오름차순 범위 재지정)
    reranged_col_idx = rerange_col_idx(idx_except_for_del_n_pad)
    row_idx = tf.cast(idx_except_for_del_n_pad[:, 0], dtype = tf.int32)
    batch_idx = tf.concat([row_idx[:, tf.newaxis], reranged_col_idx[:, tf.newaxis]], axis = -1)

    canvas_masked_n_sliced = tf.scatter_nd(indices = batch_idx, updates = val_except_for_del_n_pad, shape = tf.shape(canvas_masked) )

    return canvas_masked_n_sliced

@tf.function
def rerange_col_idx(target_batch_idx):
    '''
    col_idx의 범위를 오름차순으로 재할당해주는 함수
    '''
    
    target_col_idx = target_batch_idx[:, 1]
    
    col_idx_cumsum = tf.cast(tf.math.cumsum(target_col_idx), dtype = tf.int32)
    # col_idx_cumsum_shifted = np.append(col_idx_cumsum[1:], col_idx_cumsum[len(col_idx_cumsum)-1])
    col_idx_cumsum_shifted = tf.concat([ col_idx_cumsum[1:], col_idx_cumsum[len(col_idx_cumsum)-1][tf.newaxis] ], axis = -1)
    col_idx_diff = col_idx_cumsum_shifted - col_idx_cumsum

    # 각 row에 대해서 ceil, floor, height를 계산한 것임.
    col_idx_ceil = tf.squeeze(tf.where(col_idx_diff == 0)) + 1
    # col_idx_ceil = col_idx_ceil.numpy()

    # col_idx_floor = np.append(0, col_idx_ceil[:-1])
    col_idx_floor = tf.concat([ [0], col_idx_ceil[:-1] ], axis = -1)

    col_idx_height = col_idx_ceil - col_idx_floor
    col_idx_origin = tf.zeros(shape = tf.shape(col_idx_height), dtype = tf.int32)

    # col_idx_nested_list = tf.ragged.range(col_idx_origin, col_idx_height).to_list()
    col_idx_nested_list = tf.ragged.range(col_idx_origin, col_idx_height)
    # reranged_col_idx = tf.cast( np.concatenate(col_idx_nested_list), dtype = tf.int32)
    reranged_col_idx = tf.cast( col_idx_nested_list.merge_dims(0, -1), dtype = tf.int32)

    return reranged_col_idx

# def put_mask_n_pad_in_canvas(canvas_with_zero, batch_pad_idx, token_dict):
#     '''
#     canvas 내 [mask] 삽입 및 [pad] 재삽입 함수
#     '''

#     # 모든 0 토큰들 (= zero_buffer + zero_pad)의 인덱스에 [mask] 토큰 값 (= 9576) 삽입하기
#     batch_all_zero_idx = tf.cast(tf.where(canvas_with_zero == 0), dtype = tf.int32)
#     flatten_mask_vector = tf.ones(batch_all_zero_idx.shape[0], dtype = tf.int32) * get_token(token_dict, '[mask]')
#     canvas_with_mask = tf.tensor_scatter_nd_update(canvas_with_zero, indices = batch_all_zero_idx, updates = flatten_mask_vector)

#     # zero_pad의 인덱스 (= batch_pad_idx에 해당하는 인덱스)에 [pad] 토큰 값 (= 0) 넣어주기
#     flatten_pad_vector = tf.zeros(shape = tf.shape(batch_pad_idx)[0], dtype = tf.int32)
#     canvas_with_mask_n_pad = tf.tensor_scatter_nd_update(tensor = canvas_with_mask, indices = batch_pad_idx, updates = flatten_pad_vector)

#     return canvas_with_mask_n_pad

# def get_zero_pad_idx(inputs, canvas_with_zero):
# def get_zero_pad_idx(canvas_with_zero):
# def get_zero_pad_idx(inputs):

#     '''
#     canvas 만들면서 buffer에 삽입된 zero_buffer 를 제외한, [eos] 토큰 뒤에 등장하는 실제 [pad] 토큰 (= zero_pad)들의 idx를 뽑아내는 함수
#     '''

#     # # [pad] 토큰들 (= eos 토큰 이후의 토큰들)의 row/col 인덱싱 해주기
#     # canvas_seq_len = 2 * tf.shape(inputs)[1] - 1                                    # 캔버스 시퀀스 길이
#     # seq_len_vec = tf.repeat(canvas_seq_len, tf.shape(inputs)[0])                    # 캔버스 시퀀스 길이를 배치 갯수만큼 타일링
#     # canvas_eos_idx = indexing_eos_token(canvas_with_zero)                           # [eos] 토큰 인덱스 벡터 (= 누적합 최대값이 최초로 등장한 인덱스)

#     # # 샘플/시퀀스 별 [pad] 토큰 갯수 확보
#     # first_pad_idx = canvas_eos_idx + 1                                              # [pad] 토큰이 최초로 등장한 인덱스 (= [eos] 토큰 다음 인덱스)
#     # pad_idx_len = canvas_seq_len - tf.cast(first_pad_idx, dtype = tf.int32)         # 샘플별 [eos] 토큰 이후 등장한 [pad] 토큰들의 총 갯수

#     # # 배치 내 [pad] 토큰 인덱싱
#     # pad_col_idx_tile = np.concatenate(tf.ragged.range(first_pad_idx, seq_len_vec).to_list())[:, tf.newaxis]
#     # pad_row_idx_tile = tf.repeat(tf.range(tf.shape(inputs)[0]), pad_idx_len)[:, tf.newaxis]
#     # batch_pad_idx = tf.concat([pad_row_idx_tile, pad_col_idx_tile], axis = -1)

#     # [pad] 토큰들 (= eos 토큰 이후의 토큰들)의 row/col 인덱싱 해주기
#     seq_len = tf.shape(inputs)[1]                                    # 인풋 시퀀스 길이
#     seq_len_vec = tf.repeat(seq_len, tf.shape(inputs)[0])            # 인풋 시퀀스 길이를 배치 갯수만큼 타일링
#     eos_idx = indexing_eos_token(inputs)                             # [eos] 토큰 인덱스 벡터 (= 누적합 최대값이 최초로 등장한 인덱스)

#     # 샘플/시퀀스 별 [pad] 토큰 갯수 확보
#     first_pad_idx = eos_idx + 1                                              # [pad] 토큰이 최초로 등장한 인덱스 (= [eos] 토큰 다음 인덱스)
#     pad_idx_len = seq_len - tf.cast(first_pad_idx, dtype = tf.int32)         # 샘플별 [eos] 토큰 이후 등장한 [pad] 토큰들의 총 갯수

#     # 배치 내 [pad] 토큰 인덱싱
#     pad_col_idx_tile = np.concatenate(tf.ragged.range(first_pad_idx, seq_len_vec).to_list())[:, tf.newaxis]
#     pad_row_idx_tile = tf.repeat(tf.range(tf.shape(inputs)[0]), pad_idx_len)[:, tf.newaxis]
#     batch_pad_idx = tf.concat([pad_row_idx_tile, pad_col_idx_tile], axis = -1)

#     # 배치 내 [pad] 토큰 인덱스에 [pad] 토큰 채워넣기

#     return batch_pad_idx

@tf.function
def fill_pad_after_eos(gen_seqs, masked_prompt_inputs, token_dict):
    pad_idx = tf.cast(tf.where(masked_prompt_inputs == get_token(token_dict, '[pad]')), dtype = tf.int32)
    pad_val = tf.cast(tf.ones(shape = tf.shape(pad_idx)[0]) * get_token(token_dict, '[pad]'), dtype = tf.int32)

    padded_gen_seqs = tf.tensor_scatter_nd_update(tensor = gen_seqs, indices = pad_idx, updates = pad_val)
    return padded_gen_seqs

'''
문장 내 최초로 <eos> 토큰이 등장한 인덱스 필터링
'''
# @tf.function
def filter_first_eos_token(row_col_idx_np):

    row_col_idx_pd = pd.DataFrame(row_col_idx_np, columns=('row_idx', 'col_idx'))
    filter_row_col_idx_pd = row_col_idx_pd.groupby('row_idx').min().reset_index()
    row_col_idx_np = np.array(filter_row_col_idx_pd)

    return tf.convert_to_tensor(row_col_idx_np, dtype = tf.int32)

'''
문장 내 <eos> 토큰이 등장지점 이후를 전부 패딩
'''
# @tf.function
def padding_after_eos(gen_seqs, token_dict):
    target_row_col_idx = tf.where(tf.math.equal(gen_seqs, get_token(token_dict, '[eos]')))
    target_row_col_idx = filter_first_eos_token(target_row_col_idx)
    # pad_val = tf.cast(tf.ones(shape = tf.shape(pad_idx)[0]) * get_token(token_dict, '[pad]'), dtype = tf.int32)

    row_idx = target_row_col_idx[:, 0]                                              # 각 샘플별 sentence들의 행 인덱스
    col_idx = target_row_col_idx[:, 1]                                              # 각 샘플별 sentence가 끝나는 지점 (= sentence의 <eos> 토큰) 열 인덱스

    # [pad]가 들어가야 할 토큰들의 열 포지션
    target_token_col_pos = tf.ragged.range(col_idx+1, gen_seqs.shape[1])            # 우리의 목적은 sentence의 비정상 문장 전체를 (= <eos> 이후) 패딩하는 것임
                                                                                    # gen_seqs.shape[1]은 패딩이 적용될 대상 배치의 최대길이 문장의 길이
                                                                                    # 즉, range(col_idx+1, gen_seqs.shape[1])은 col_idx 다음부터 마지막 토큰 index까지 고려한다는 의밍미

    # ragged 포맷의 데이터를 flatten 시키는 트릭
    target_token_col_pos = tf.cast(target_token_col_pos, dtype = tf.int32)
    target_token_col_pos_flat = np.concatenate(target_token_col_pos.to_list())

    # [pad]가 들어가야 할 토큰들의 행 포지션
    target_token_row_pos = tf.repeat(row_idx, (gen_seqs.shape[1]-col_idx-1))        # 행 인덱스 계산시 <eos> 토큰 등장지점 "이후"를 전부 고려해야 함
                                                                                    # col_idx "이후" 부터 고려했으므로 col_idx에 해당하는 토큰을 count에서 제외하기 위해 -1 해주기 

    # 패딩이 적용될 최종 타겟 포지션들의 행열 인덱스                                                                                   
    final_target_pos_idx = tf.concat([target_token_row_pos[:, tf.newaxis], target_token_col_pos_flat[:, tf.newaxis]], axis = 1)

    # 패딩을 적용하여 업데이트
    pad_val = tf.cast(tf.ones(shape = tf.shape(final_target_pos_idx)[0]) * get_token(token_dict, '[pad]'), dtype = tf.int32)
    padded_gen_seqs = tf.tensor_scatter_nd_update(tensor = gen_seqs, indices = final_target_pos_idx, updates = pad_val)

    return padded_gen_seqs

@tf.function
def return_action_token(agent_outputs, inputs, token_dict, mode = 'train'):
    '''
    agent_ouputs을 받아 token으로 변환해주는 함수
    '''

    # action의 종류가 [INS_F], [INS_B], [INS_A], [DEL], [REP], [KEP] 으로 그 크기가 6인 경우를 생각해보자.
    # 이 경우, agent가 반환하는 action의 토큰 값은 [0, 1, 2, 3, 4, 5] 이다.
    # 위와 같은 토큰 값을 그대로 사용할 경우 기존 token_dict 내에서 위의 값을 가지는 토큰들과 충돌한다.
    # 따라서, 이 문제를 해결하기 위해 다음과 같은 절차를 따른다.
    # (1) 먼저, token_dict에 상기의 6가지 action을 차례대로 token_dict의 끝에서부터 추가해준다.
    # (2) 6가지 action 중 가장 처음 나오는 action (e.g., 위의 경우 [INS_F])의 token_dict에서의 토큰값을 얻는다
    # (3) 얻은 토큰값을 [0, 1, 2, 3, 4, 5] 에 그대로 broadcast하여 더해준다.
    # (4) 이에 따라, agent가 반환하는 action의 토큰 값은 기존의 token_dict의 마지막 6개 토큰 값을 갖게된다.


    # stochastic-selection when train mode
    if mode == 'train':
        agent_actions = tf.squeeze(tfp.distributions.Categorical(logits = agent_outputs).sample(1))
    
    # greedy-selection when test mode
    elif mode == 'test':
        agent_actions = tf.argmax(agent_outputs, axis = -1)
    pad_mask = tf.cast(tf.math.not_equal(inputs, get_token(token_dict, '[pad]')), dtype = tf.int32)
    action_tokens = tf.cast(agent_actions, dtype = tf.int32) + get_token(token_dict, '[INS_F]')
    action_tokens = tf.math.multiply(action_tokens, pad_mask)

    return agent_actions, action_tokens

def split_by_polarity(masked_inputs, polars):

    pos_idx = tf.where(polars == 1)
    # pos_masked_inputs = masked_inputs[pos_idx, :]
    pos_masked_inputs = tf.gather_nd(params = masked_inputs, indices = pos_idx)

    neg_idx = tf.where(polars == 0)
    # neg_masked_inputs = masked_inputs[neg_idx, :]
    neg_masked_inputs = tf.gather_nd(params = masked_inputs, indices = neg_idx)

    return pos_masked_inputs, pos_idx, neg_masked_inputs, neg_idx

# def concat_reward_n_sort_back_idx(reward1, pos_idx, reward2, neg_idx):
#     '''
#     리워드 두개를 합쳐 concat_reward 생성
#     이후, concat_reward의 행을 pos_idx와 neg_idx에 대해 오름차순 정렬
#     '''

#     concat_idx = tf.squeeze(tf.concat([pos_idx, neg_idx], axis = 0))
#     argsorted_idx = tf.argsort(concat_idx)

#     concat_reward = tf.concat([reward1, reward2], axis = 0)

#     concat_reward_row_sorted = tf.gather(concat_reward, indices = argsorted_idx, axis = 0)     # concat_reward의 행을 inputs의 원래 행 순서대로 복원.

#     return concat_reward_row_sorted

def concat_n_sort_back_idx(a, a_idx, b, b_idx):
    '''
    두 행렬 a, b를 합친 후 원래 idx 순서에 맞춰 오름차순 재정렬하는 함수
    '''

    concat_idx = tf.squeeze(tf.concat([a_idx, b_idx], axis = 0))
    argsorted_idx = tf.argsort(concat_idx)

    concat_ab = tf.concat([a, b], axis = 0)

    concat_ab_row_sorted = tf.gather(concat_ab, indices = argsorted_idx, axis = 0)     # concat_ab의 행을 inputs의 원래 행 순서대로 복원.

    return concat_ab_row_sorted

def get_num_of_kep(action_tokens, inputs, token_dict):
# def get_num_of_kep(action_tokens, token_dict):

    '''
    [KEP] 토큰 갯수 세기
    '''

    kep_idx = tf.math.equal(action_tokens, get_token(token_dict, '[KEP]'))
    non_bos_idx = tf.math.not_equal(action_tokens, get_token(token_dict, '[bos]'))
    non_eos_idx = tf.math.not_equal(action_tokens, get_token(token_dict, '[eos]'))
    kep_final_idx = kep_idx & (non_bos_idx & non_eos_idx)
    kep_onehot = tf.cast(kep_final_idx, dtype = tf.float32)

    mean_kep_score = tf.reduce_sum(kep_onehot, axis = -1)/tf.cast(tf.shape(inputs)[1], dtype = tf.float32)
    kep_score = -tf.math.log(1-mean_kep_score)
    # kep_score = -tf.math.log(1-tf.reduce_mean(kep_onehot, axis = -1))
    return kep_score

@tf.function
def get_num_of_ops(action_tokens, inputs, token_dict, ops_token):

    '''
    지정한 Operation 토큰 (e.g., [KEP], [REP], [DEL] 등) 갯수 세기
    '''

    ops_idx = tf.math.equal(action_tokens, get_token(token_dict, ops_token))
    non_bos_idx = tf.math.not_equal(action_tokens, get_token(token_dict, '[bos]'))
    non_eos_idx = tf.math.not_equal(action_tokens, get_token(token_dict, '[eos]'))
    non_pad_idx = tf.math.not_equal(action_tokens, get_token(token_dict, '[pad]'))

    active_idx = (non_bos_idx & non_eos_idx) & non_pad_idx
    active_onehot = tf.cast(active_idx, dtype = tf.float32)

    active_ops_idx = ops_idx & active_idx
    active_ops_onehot = tf.cast(active_ops_idx, dtype = tf.float32)

    # mean_ops_score = tf.reduce_sum(active_ops_onehot, axis = -1)/tf.cast(tf.shape(inputs)[1], dtype = tf.float32)
    mean_ops_score = tf.reduce_sum(active_ops_onehot, axis = -1)/tf.reduce_sum(active_onehot, axis = -1)
    # ops_score = -tf.math.log(1-mean_ops_score)
    # return ops_score
    return mean_ops_score

@tf.function
def get_len_of_seq(inputs, token_dict):

    '''
    [KEP] 토큰 갯수 세기
    '''

    non_bos_idx = tf.math.not_equal(inputs, get_token(token_dict, '[bos]'))
    non_eos_idx = tf.math.not_equal(inputs, get_token(token_dict, '[eos]'))
    non_pad_idx = tf.math.not_equal(inputs, get_token(token_dict, '[pad]'))
    token_idx = non_pad_idx & (non_bos_idx & non_eos_idx)
    token_onehot = tf.cast(token_idx, dtype = tf.float32)

    total_token_num = tf.reduce_sum(token_onehot, axis = -1)
    return total_token_num


def match_seqlen_equal(inputs, masked_inputs):
    '''
    inputs의 시퀀스 길이와 masked_inputs의 시퀀스 길이를 맞춰주기 위해,
    inputs와 masked_inputs의 시퀀스 길이 차이 만큼의 pad 행렬을 inputs 뒤에 붙여주는 함수
    '''
    pad_mat = tf.zeros(shape = (tf.shape(masked_inputs)[0], tf.shape(masked_inputs)[1] - tf.shape(inputs)[1]), dtype=tf.int32) 
    padded_inputs = tf.concat([inputs, pad_mat], axis = -1)

    return padded_inputs

@tf.function
def get_action_tokens_staistics(action_tokens, token_dict):
    '''
    디폴트 토큰이 필요함 -> 없으면 reward 저장이 안될 때가 있음.
    '''
    edit_token_list = ['[INS_F]', '[INS_B]', '[INS_A]', '[DEL]', '[REP]', '[KEP]']
    default_action_tokens = [get_token(token_dict, t) for t in edit_token_list]
    default_action_tokens = tf.reshape(tf.convert_to_tensor(default_action_tokens, dtype = tf.int32), shape = (1, -1))

    concat_action_tokens = tf.concat([default_action_tokens, tf.reshape(action_tokens, shape= (1, -1))], axis = -1)

    concat_action_tokens_flat = tf.squeeze(concat_action_tokens)
    action_types, _, action_freqs = tf.unique_with_counts(concat_action_tokens_flat)
    sort_idx = tf.argsort(action_types)
    sorted_action_types = tf.gather(action_types, sort_idx)
    sorted_action_freqs = tf.gather(action_freqs, sort_idx)

    return sorted_action_types, sorted_action_freqs

'''
토크나이징 된 벡터를 원형값으로 디코딩하는 함수
'''
def decodes_token(token_vector, token_dict):
    '''
    토큰 사전의 value 벡터를 넘파이 배열로 정의
    '''
    token_dict_values = np.array(list(token_dict.values()))

    '''
    토크나이징 된 벡터의 모든 (row_idx, col_idx) 조합을 뽑아서 long-format으로 구축하는 작업
    '''
    row_idx = tf.range(tf.shape(token_vector)[0])
    col_idx = tf.range(tf.shape(token_vector)[1])
    L_ = len(col_idx)   # L_ : 시퀀스 길이
    N_ = len(row_idx)   # N_ : 샘플 갯수
    row_idx_repeat = tf.repeat(row_idx, L_)[:, tf.newaxis]                # 각 row_idx를 시퀀스 길이 (= L_)만큼 복제
    col_idx_tile = tf.tile(col_idx, multiples = (N_, ))[:, tf.newaxis]    # 전체 col_idx를 샘플 갯수 (= N_)만큼 타일링
    target_idx = tf.concat([row_idx_repeat, col_idx_tile], axis = -1)

    '''
    모든 (row_idx, col_idx) 조합에 해당하는 토큰 값들을 수집 (gather)하여 1차원 벡터형태인 target_vals로 정의
    '''
    target_vals = tf.gather_nd(params = token_vector, indices = target_idx)

    '''
    위에서 구한 (기존의 토크나이징 된 벡터를 flatten 시킨) target_vals를 index로 간주하고, 
    token_dict_values에서 위 index (= target_vals)에 해당하는 value를 가져오기. 이 때 이 value는 토큰이 아닌 원형의 값 (예., 단어)임.
    
    즉, 위의 과정을 통해 우리는 토큰 -> 원형의 값으로 디코딩하는 것이며, 
    이 디코딩된 결과는 1차원 벡터이므로 reshape를 통해 원래 token_vector의 형태로 복원해준다.
    '''
    decoded_token_vector = tf.reshape(tf.gather(params = token_dict_values, indices = target_vals), shape = (tf.shape(token_vector)[0], -1))

    return decoded_token_vector

'''
토큰 벡터를 디코딩 후  리스트화 하는 함수
'''
def get_decoded_list(token_vector, token_dict):

    '''
    토큰 벡터를 원형태로 디코딩
    '''
    decoded_vector = decodes_token(token_vector, token_dict)

    '''
    토큰 벡터에서, '[pad]', '[bos]', '[eos]'에 해당하는 토큰은 제외하기.
    '''
    pad_token = get_token(token_dict, '[pad]')
    bos_token = get_token(token_dict, '[bos]')
    eos_token = get_token(token_dict, '[eos]')
    target_idx = tf.cast(tf.where(tf.math.logical_and(tf.math.not_equal(token_vector, pad_token), tf.math.logical_and(tf.math.not_equal(token_vector, bos_token), tf.math.not_equal(token_vector, eos_token)))), dtype = tf.int32)
    target_vals = tf.gather_nd(params = decoded_vector, indices = target_idx)
    decoded_vector_np = tf.scatter_nd(indices = target_idx, updates = target_vals, shape = tf.shape(decoded_vector)).numpy().astype('str')       

    '''
    디코딩 된 벡터를 리스트화 하기
    '''
    decoded_vector_list = decoded_vector_np.tolist()
    decoded_vector_list = [list(filter(None, each_vec)) for each_vec in decoded_vector_list]

    return decoded_vector_list


'''
코사인-감쇄-재시작 스케쥴러
'''
def CosineDecayRestart(an_epoch, max_eta, decay_step = 25, m_mul = 0.9, min_eta = 0.001):

    pi = tf.cast(np.pi, dtype = tf.float32)
    cur_step = an_epoch % decay_step

    if cur_step == 0:
        if an_epoch != 0:
            max_eta = m_mul * max_eta
        
    new_eta = min_eta + 0.5 * (max_eta - min_eta) * (1 + tf.math.cos( (cur_step * pi) / decay_step ) )

    return new_eta, max_eta


'''
reposition이 알려진 compound의 original/secondary 타겟 시퀀스 및-indication code 페어 생성
'''
def get_reposition_pair(drug_reposition_dat, disease_code_map):

    reposition_disease_pair = []
    reposition_code_pair = []
    ori_reposition_seq_code_pair = []
    sec_reposition_seq_code_pair = []
    for tmp_drug in np.unique(drug_reposition_dat['Related drug'].tolist()):
        target_drug = drug_reposition_dat[drug_reposition_dat['Related drug'] == tmp_drug]
        
        # Original/Secondary Indication의 disease 및 disease_code 리스트 생성
        ori_ind_disease = target_drug[target_drug['Indication'] == 'Original']['Related disease'].tolist()
        sec_ind_disease = target_drug[target_drug['Indication'] == 'Secondary']['Related disease'].tolist()
        ori_ind_disease_code = np.where(np.array(list(disease_code_map.values())) == ori_ind_disease)[0][0]
        sec_ind_disease_code = np.where(np.array(list(disease_code_map.values())) == sec_ind_disease)[0][0]
        print('original disease : {}, code : {}'.format(ori_ind_disease, ori_ind_disease_code))
        print('secondary disease : {}, code : {}'.format(sec_ind_disease, sec_ind_disease_code))

        # Reposition (original-secondary) 관계를 묶은 disease 및 disease_code 리스트 생성
        reposition_disease_pair += [[ori_ind_disease + sec_ind_disease]]
        reposition_code_pair += [[ori_ind_disease_code, sec_ind_disease_code]]

        # Original Target Disease와 Secondary Indication disease code를 묶은 리스트 생성
        ori_target_disease = target_drug[target_drug['Indication'] == 'Original']['top'].tolist()
        sec_target_disease = target_drug[target_drug['Indication'] == 'Secondary']['top'].tolist()
        ori_reposition_seq_code_pair += [[ori_target_disease[0], ori_ind_disease_code]]
        sec_reposition_seq_code_pair += [[sec_target_disease[0], sec_ind_disease_code]]

    return reposition_disease_pair, reposition_code_pair, ori_reposition_seq_code_pair, sec_reposition_seq_code_pair

'''
1) 주어진 단백질 시퀀스 (protein_seq)를 기준으로 interaction_dat을 filtering 하고
2) 해당 단백질 시퀀스가 관련된 disease의 indication code 컬럼을 생성
'''
def filter_interaction_dat(protein_seq, ind_code, interaction_dat):

    ind_code_list = []
    for i in range(len(protein_seq)):
        if i == 0:
            # interaction_dat 필터링
            target_seq_filttered_dat = interaction_dat[interaction_dat['Target_seq'] == protein_seq[i]]
            target_interaction_dat = target_seq_filttered_dat

            # indication_code 컬럼 생성
            ind_code_list += [ind_code[i]] * target_seq_filttered_dat.shape[0]

        else:
            target_seq_filttered_dat = interaction_dat[interaction_dat['Target_seq'] == protein_seq[i]]
            target_interaction_dat = pd.concat([target_interaction_dat, target_seq_filttered_dat], axis = 0)

            # indication_code 컬럼 생성
            ind_code_list += [ind_code[i]] * target_seq_filttered_dat.shape[0]

    # Original Indication에 해당하는 Code 컬럼 추가해주기
    target_interaction_dat = target_interaction_dat[['Related drug', 'Related disease', 'Protein ID', 'Sequence', 'Compound_smiles']]


    return target_interaction_dat, ind_code_list

'''
ddd
'''
def get_second_ind_code_vector(ori_interaction_dat, ori_ind_code_list, code_reposition_map):

    sec_ind_array = np.ones(shape = len(ori_ind_code_list))
    for key, val in dict(code_reposition_map).items():
        # Original Indication이 3인 경우 Secondary Indication이 5 또는 1이 될 수 있음.
        # Drug name이 "Topiramate'일 경우 5, 'Valproic acid'일 경우 1임
        if key == 3:
            target_idx = np.where(np.array(ori_ind_code_list) == key)[0]
            drug_name_array = np.array(ori_interaction_dat['Related drug'])[target_idx]
            ori_ind_3_1 = np.where(drug_name_array == 'Topiramate')[0]
            ori_ind_3_2 = np.where(drug_name_array == 'Valproic acid')[0]
            sec_ind_array[target_idx[ori_ind_3_1]] = int(5)
            sec_ind_array[target_idx[ori_ind_3_2]] = int(1)
        else:
            sec_ind_array[np.where(np.array(ori_ind_code_list) == key)[0]] = int(val)

    return sec_ind_array.astype(np.int32)


'''
synthetic_data.py 파일에 사용될 분석용 함수코드
'''
from scipy.stats import norm, truncnorm, truncexpon, poisson

def get_truncated_normal(mean=0, std=1, low=1, upp=10):
    '''
    truncated 정규분포 정의
    '''
    return truncnorm(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std)

def get_truncated_poisson(num_sample, mean=None):

    trunc_sample = np.zeros([1, 0])

    while True:

        # 람다 (lambda)가 1인 포아송 분포로부터 num_sample만큼 샘플링
        poisson_tmp_sample = poisson.rvs(mu=mean, size=num_sample)
        trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        # 1 ~ 10 사이만 필터링
        trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        # # 좌측으로 치우쳐진 포아송 분포
        # if side == 'left':

        #     # 람다 (lambda)가 1인 포아송 분포로부터 num_sample만큼 샘플링
        #     poisson_tmp_sample = poisson.rvs(mu=mean, size=num_sample)
        #     trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        #     # 1 ~ 10 사이만 필터링
        #     trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        #     trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        # # 우측으로 치우쳐진 포아송 분포
        # elif side == 'right':

        #     # 람다 (lambda)가 20인 포아송 분포로부터 num_sample만큼 샘플링
        #     # 원래 의도는 람다가 10인 분포를 뽑으려고 한건데, 포아송 분포 특성상 10으로 하니 람다=1일 때와 대칭적인 모습이 나오지 않음.
        #     # 따라서, 람다=20으로 해서 더 right-skewed 만들어 람다=1일 때와 대칭적인 모습이 보이도록 강제.
        #     poisson_tmp_sample = poisson.rvs(mu=20, size=num_sample)
        #     trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        #     # 1 ~ 10 사이만 필터링
        #     trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        #     trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        print('leN_turnc_sample : {}'.format(len(trunc_sample)))
        if len(trunc_sample) >= num_sample:
            return trunc_sample[:num_sample]

def get_samples(num_sample, mu=None, sigma=None, dist='truncnorm', sample_ratio=None, mu1=None, mu2=None, sigma1=None, sigma2=None):

    # x축이 특정 구간으로 절삭된 정규분포 정의
    if dist == 'truncnorm':

        # truncated 정규분포 정의
        trunc_normal_dist = get_truncated_normal(mean=mu, std=sigma, low=1, upp=10)

        # truncated 정규분포로부터 행동 샘플링
        samples = trunc_normal_dist.rvs(num_sample)
        integer_samples = np.round(samples).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        # 행동 별 발생빈도 추출
        actions, counts = np.unique(integer_samples, return_counts=True)

    # x축이 특정 구간으로 절삭된 포아송 분포 정의
    elif dist == 'truncpoisson':
        samples = get_truncated_poisson(num_sample, mean=mu)
        integer_samples = samples.astype('int32')
        actions, counts = np.unique(integer_samples, return_counts=True)

    # x축이 특정 구간으로 절삭된 양봉 정규분포 정의
    elif dist == 'trunc_doublenorm':

        # 양봉 truncated 정규분포 정의
        # 두번째 모드의 행동정책 분포의 평균이 첫번째 행동정책 분포의 평균보다 4 더 크게 설정
        trunc_normal_dist1 = get_truncated_normal(mean=mu1, std=sigma1, low=1, upp=10)
        trunc_normal_dist2 = get_truncated_normal(mean=mu2, std=sigma2, low=1, upp=10)

        # 양봉 truncated 정규분포로부터 행동 샘플링
        # 첫번째 모드의 행동정책 분포의 샘플 비율이 전체 샘플의 8할, 두번째 모드가 2할을 포함하도록 설정
        sample1_ratio = sample_ratio
        samples1 = trunc_normal_dist1.rvs(int(num_sample * sample1_ratio))
        integer_samples1 = np.round(samples1).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        samples2 = trunc_normal_dist2.rvs(num_sample - int(num_sample * sample1_ratio))
        integer_samples2 = np.round(samples2).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        # 행동 별 발생빈도 추출
        integer_samples = np.append(integer_samples1, integer_samples2)
        actions, counts = np.unique(integer_samples, return_counts=True)


    return actions, counts, integer_samples

def reward_function(x, order=2):

    # 차수
    rewards = x**order

    # # 누적합이 1이 되도록 소프트맥스 정규화
    # softmax_rewards = np.exp(rewards)/np.exp(rewards).sum()
    # softmax_rewards = np.exp(rewards)
    rewards = rewards/rewards.sum()

    # reward의 값의 범위가 [0, 1]이 되도록 min-max 정규화
    rewards = (rewards - np.min(rewards)) /(np.max(rewards) - np.min(rewards))

    return rewards

def get_reward_dist(min_action, order):
    '''
    보상 샘플추출
    - min_action : reward가 반환되는 최소 절삭 action
    - order : reward 함수의 차수
    '''

    x = np.arange(1, 11, 1)

    try:
        x_new = x[x>=min_action]
        x[x<min_action] = 0
        x[x>=min_action] = x_new

    except:
        pass;

    # 보상계산
    reward_dist = reward_function(x, order=order)

    return reward_dist

def get_rewards(onehot_action, reward_dist):
    
    # reward = tf.reduce_sum(tf.multiply(reward_dist, onehot_action))
    reward = tf.matmul(onehot_action, reward_dist[:, tf.newaxis])

    return reward


def get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs):

    # 파라미터 설정
    action_size = kwargs['action_size']
    epi_len = kwargs['epi_len']
    num_cases = kwargs['num_cases']

    # 누적 에피소딕 상태표현의 synthetic dataset 구축
    sampled_actions_data = synthetic_data['bp_sampled_actions'] - 1
    synthetic_onehot_states = tf.one_hot(sampled_actions_data, depth = action_size)                                         # synthetic_onehot_states: (total_sample_size, action_size)
    synthetic_onehot_episodic_states = tf.reshape(synthetic_onehot_states, shape = (num_cases, -1, epi_len, action_size))             # synthetic_onehot_episodic_states : (num_cases, per_case_num_epi, epi_len, action_size)
    synthetic_cum_episodic_states = tf.cumsum(synthetic_onehot_episodic_states, axis = 2)                                       # 에피소드 길이동안 visit frequency 누적

    # 초기 상태 (init_state) 벡터 고려 (epi_len + 1 해주기)
    shape_vec = tf.shape(synthetic_cum_episodic_states).numpy()     # synthetic_cum_episodic_states 형상벡터 추출
    shape_vec[2] = 1                                                # episode축의 차원을 1차원으로 설정
    init_state = tf.zeros(shape = shape_vec)                        # episode의 초기 상태 벡터 정의
    synthetic_cum_episodic_states = tf.concat([init_state, synthetic_cum_episodic_states], axis=2)      # episode축을 따라 초기 상태 벡터 prepend
                                                                                                        # (num_cases, per_case_num_epi, epi_len + 1, action_size)

    # input data (= state_history) 구축
    cur_state = synthetic_cum_episodic_states[:, :, :-1, :]                             # state_history : (num_cases, per_case_num_epi, epi_len + 1, action_size)
                                                                                        # episode의 마지막 state에서는 action sampling이 아니라 종료가 수행됨.
                                                                                        # 즉, 마지막 state에 해당하는 부분은 학습데이터로 고려해주지 않아됨.
                                                                                        # 따라서, 마지막 state를 뺀 나머지 부분들에 대해서 cur_state 정의
                                                                                        # cur_state : (num_cases, per_case_num_epi, epi_len, action_size)
    input_data = copy.deepcopy(cur_state)                                               # input_data : (num_cases, per_case_num_epi, epi_len, action_size)

    # target data (= action history)구축
    target_data = tf.reshape(sampled_actions_data, shape = (num_cases, -1, epi_len, 1))       # target_data : (num_cases, per_case_num_epi, epi_len, 1)

    # reward data (= reward history) 구축
    next_state = copy.deepcopy(synthetic_onehot_episodic_states)        # reward는 next_state에 의해 결정되며, next_state는 action에 의해 정의됨
                                                                        # next_state : (num_cases, per_case_num_epi, epi_len, action_size)
    reward_data = get_rewards(next_state, reward_dist)

    # case 별 분리
    # num_cases = synthetic_cum_episodic_states.shape[0]
    input_data_by_case = [input_data[i, :, :, :] for i in range(num_cases)]        # 케이스 별로 인풋 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)
    target_data_by_case = [target_data[i, :, :, :] for i in range(num_cases)]      # 케이스 별로 타겟 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)
    reward_data_by_case = [reward_data[i, :, :, :] for i in range(num_cases)]      # 케이스 별로 보상 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)

    return input_data_by_case, target_data_by_case, reward_data_by_case


'''
데이터 사이즈를 특정 비율로 자르기
'''
def truncate_datasize_by_ratio(data, ratio:float=None):
    '''
    ratio : (0, 1]
    '''

    # 전체 데이터셋에서 주어진 ratio에 해당하는 만큼의 샘플 갯수 정의
    num_samples = int(data.shape[0] * ratio)

    # 전체 데이터셋에 대한 인덱스 정의
    all_idx = tf.range(0, data.shape[0])

    # 전체 데이터셋 셔플링 후 ratio에 해당하는 만큼의 샘플 갯수만 추출
    truncation_idx = tf.random.shuffle(all_idx)[:num_samples]

    # 인덱스 소팅 
    truncation_idx = tf.sort(truncation_idx)

    return truncation_idx

def get_truncated_data(data, truncation_idx):
    trunc_data = tf.gather(params=data, indices=truncation_idx, axis=0)

    return trunc_data

'''
주어진 프롬프트 길이 내애 <pad>가 존재하는 경우 인덱싱 및 제거
'''
def indice_pad_in_prefix(prefix_data, prefix_len:int, pad_token_id:int):

    # 프롬프트 내에 <pad>가 존재하지 않는 경우 hit
    target_idx = tf.math.not_equal(prefix_data[:, :prefix_len], pad_token_id)
    target_idx = tf.cast(target_idx, dtype=tf.int32)

    # 샘플별 hit의 총합이 프롬프트 길이와 같은 경우만 인덱싱
    # 즉, 프롬프트 내에 <pad>가 존재하지 않는 경우 인덱싱
    target_idx = tf.where(tf.reduce_sum(target_idx, axis = 1) == prefix_len)

    return target_idx

def remove_pad_in_prefix_case(target_idx, target_data):

    # target_idx를 활용해 target_data 인덱싱
    indiced_target_data = tf.squeeze(tf.gather(target_data, target_idx))

    return tf.reshape(indiced_target_data, shape = (-1, target_data.shape[1]))

'''
eos_token이 우연이 제일 마지막 토큰으로 또는 아예 안나오는 경우에 대해서
- case 1: Last index -> Last index + 1 -> 1st index
- case 2: None -> 1st index -> Last index
의 두 경우를 순차적으로 수행하여, 전부 Last index 으로 처리
'''
def eos_index_change(gen_inputs, after_eos_token_col_idx, eos_idx_of_interest):

    '''
    --------------------------------------------------------------------------------------------------
    case 1 : eos_idx_of_interest = tf.shape(gen_inputs)[1] + 1
    
    - eos_token_idx : Last = tf.shape(gen_inputs)[1]-1 --(+1)--> tf.shape(gen_inputs)[1] -> 1
    - eos_idx_of_interest : tf.shape(gen_inputs)[1]
    - eos index change : tf.shape(gen_inputs)[1] -> 1

    - summary : Last index -> Last index + 1 -> 1st index
    - return : after_eos_token_col_idx_updated

    --------------------------------------------------------------------------------------------------
    case 2 : eos_idx_of_interest = 1

    - eos_token_idx : None -> 0 --(+1)--> 1 -> tf.shape(gen_inputs)[1]-1
    - eos_idx_of_interest : 1
    - eos index change : 1 -> tf.shape(gen_inputs)[1]-1

    - summary : None -> 1st index -> Last index
    - return : after_eos_token_col_idx_updated
    --------------------------------------------------------------------------------------------------
    '''

    # 생성결과 (gen_inputs)에 eos_idx_of_interest 이 존재하는 경우, 해당 생성결과의 row_idx 뽑기
    target_row_idx = tf.where(tf.math.equal(after_eos_token_col_idx, eos_idx_of_interest))


    # (case 1)의 경우
    # - eos_token이 마지막 시점에 등장한 (after_eos_token_col_idx = tf.shape(gen_inputs).shape[1]+1 인) row_idx들에 대해서,
    # - eos_token이 등장한 col_idx 값을 1st 시점 (= 1) 로 바꿔주기 위해
    # - 1 값을 target_row_idx 갯수만큼 복제
    # - 이 복제된 eos_update_idx 는 after_eos_token_col_idx = tf.shape(gen_inputs).shape[1]+1 인 부분 (= target_row_idx)을
    # - 1 값으로 update
    if eos_idx_of_interest != 1:
        eos_update_idx = tf.repeat(1, tf.shape(target_row_idx)[0])
        after_eos_token_col_idx2 = tf.tensor_scatter_nd_update(tensor=after_eos_token_col_idx, 
                                                        indices=target_row_idx, 
                                                        updates=eos_update_idx)
    # (case 2)의 경우
    # - eos_token이 존재하지 않은 (after_eos_token_col_idx = 1 인) row_idx들에 대해서,
    # - eos_token이 등장한 col_idx 값을 마지막 시점 (= tf.shape(gen_inputs)[1]-1) 로 바꿔주기 위해
    # - tf.shape(gen_inputs)[1]-1 값을 target_row_idx 갯수만큼 복제
    # - 이 복제된 eos_update_idx 는 after_eos_token_col_idx = 1 인 부분 (= target_row_idx)을
    # - tf.shape(gen_inputs)[1]-1 값으로 update
    else:
        eos_update_idx = tf.repeat(tf.shape(gen_inputs)[1]-1, tf.shape(target_row_idx)[0])
        after_eos_token_col_idx2 = tf.tensor_scatter_nd_update(tensor=after_eos_token_col_idx, 
                                                        indices=target_row_idx, 
                                                        updates=eos_update_idx)


    return after_eos_token_col_idx2


'''
eos_token 직후 (aka right-direction)의 모든 토큰들을 패딩
'''
def right_pad_after_eos_token(gen_inputs, eos_token_id:int, pad_token_id:int, total_len:int):

    '''
    모든 샘플에 대해서 eos_token 직후 첫 컬럼위치를 인덱싱 = after_eos_token_col_idx
    '''
    eos_token_hit_matrix = tf.math.equal(gen_inputs, eos_token_id)

    # eos는 포함시키면 안되고, 그 뒤 시점부터 padding의 대상임. 따라서, eos가 등장한 시점인 argmax의 이후 시점을 인식하도록 +1 해주어야 함.
    after_eos_token_col_idx = tf.argmax(eos_token_hit_matrix, axis = -1).numpy() + 1

    '''
    eos_token 위치 업데이트 첫번째 (case 1)
    - eos_token이 Last index에 등장한 샘플의 경우 (= after_eos_token_col_idx = tf.shape(gen_inputs)[1]+1)에 대한 경우 
    - eos_token_idx : Last index -> Last index + 1 -> 1st index
    '''
    after_eos_token_col_idx_updated = eos_index_change(gen_inputs,
                                                       after_eos_token_col_idx = after_eos_token_col_idx,
                                                       eos_idx_of_interest = tf.shape(gen_inputs)[1])

    '''
    eos_token 위치 업데이트 두번째 (case 2)
    - eos_token이 존재하지 않은 샘플의 경우 (= after_eos_token_col_idx = 1)에 대한 경우
    - eos_token_idx : None -> 1st index -> Last index
    '''
    after_eos_token_col_idx_updated = eos_index_change(gen_inputs,
                                                       after_eos_token_col_idx = after_eos_token_col_idx_updated,
                                                       eos_idx_of_interest = 1)


    '''
    최종 eos_token 이후 토큰들의 인덱스 (= after_eos_token_col_idx_updated) 를 가지고 padding 수행
    '''
    # 업데이트 된 after_eos_token_col_idx (= after_eos_token_col_idx2)로 col/row idx 추출
    after_eos_token_col_idx3 = tf.ragged.range(after_eos_token_col_idx_updated, total_len)
    after_eos_token_row_idx = [np.repeat(i, len(val)) for i, val in enumerate(after_eos_token_col_idx3)]

    col_idx = np.concatenate(list(after_eos_token_col_idx3))[:, np.newaxis]
    row_idx = np.concatenate(after_eos_token_row_idx)[:, np.newaxis]
    target_idx = np.concatenate([row_idx, col_idx], axis = -1)

    pad_matrix = tf.ones(shape=tf.shape(target_idx)[0], dtype = tf.int32) * pad_token_id

    gen_inputs_right_pad = tf.tensor_scatter_nd_update(tensor=gen_inputs, indices=target_idx, updates=pad_matrix)

    return gen_inputs_right_pad

# %%
