# %%
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFBertModel

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
OpenAI GPT2 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# OpenAI GPT2 토크나이저 저장 및 호출경로 생성
GPT2_small_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_small'
if os.path.exists(GPT2_small_SAVE_PATH):
    print(f"{GPT2_small_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(GPT2_small_SAVE_PATH, exist_ok=True)
    print(f"{GPT2_small_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # GPT2_small 좌측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    GPT2_small_tokenizer.save_pretrained(GPT2_small_SAVE_PATH + '/tokenizer_left')

    # GPT2_small 우측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    GPT2_small_tokenizer.save_pretrained(GPT2_small_SAVE_PATH + '/tokenizer_right')

    # GPT2_small 모델 임포트
    GPT2_small_model = TFAutoModelForCausalLM.from_pretrained("gpt2")

    # GPT2_small 모델 저장
    GPT2_small_model.save_pretrained(GPT2_small_SAVE_PATH + '/model')


'''
MIT GPT2-large 토크나이저 및 모델 가중치 저장 및 로드
- Behavior Policy로써 기능할 Large LM 모델
'''
# MIT GPT2-large 토크나이저 저장 및 호출경로 생성
GPT2_large_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_large'
if os.path.exists(GPT2_large_SAVE_PATH):
    print(f"{GPT2_large_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(GPT2_large_SAVE_PATH, exist_ok=True)
    print(f"{GPT2_large_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'

    # GPT2_large 좌측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2-large",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='left')
    GPT2_small_tokenizer.save_pretrained(GPT2_large_SAVE_PATH + '/tokenizer_left')

    # GPT2_large 우측패딩 토크나이저 저장
    GPT2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2-large",
                                                    bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                    padding="max_length", truncation=True, padding_side='right')
    GPT2_small_tokenizer.save_pretrained(GPT2_large_SAVE_PATH + '/tokenizer_right')

    # GPT2_large 모델 임포트
    GPT2_large_model = TFAutoModelForCausalLM.from_pretrained("gpt2-large")

    # GPT2_large 모델 저장
    GPT2_large_model.save_pretrained(GPT2_large_SAVE_PATH + '/model')


'''
BERT 토크나이저 및 모델 가중치 저장 및 로드
- Reference 샘플과 Bootstarap 샘플간 Similarity 계산 모델
'''
# BERT 토크나이저 저장 및 호출경로 생성
BERT_SAVE_PATH = parent_dir + '/pretrained_weights/bert'
# BERT_SAVE_PATH = parent_dir + '/pretrained_weights1/BERT'
if os.path.exists(BERT_SAVE_PATH):
    print(f"{BERT_SAVE_PATH} -- Folder already exists \n")

else:
    os.makedirs(BERT_SAVE_PATH, exist_ok=True)
    print(f"{BERT_SAVE_PATH} -- Folder create complete \n")

    '''
    최초의 토크나이저를 허깅페이스로부터 로드하여 로컬에 저장
    '''
    # BERT 토크나이저 로드
    BOS = '<bos>'
    EOS = '<eos>'
    MASK = '[MASK]'
    PAD = '<pad>'
    SEP = '</s>'
    CLS = '[CLS]'
    # TFBert 토크나이저 임포트
    # BERT_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
    #                                                 bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, cls_token = CLS, mask_token = MASK, sep_token = SEP,
    #                                                 padding="max_length", truncation=True)
    BERT_Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                    unk_token='<unk>', pad_token=PAD, 
                                                    cls_token = BOS, mask_token = MASK, sep_token = EOS,
                                                    padding="max_length", truncation=True)
    BERT_Tokenizer.save_pretrained(BERT_SAVE_PATH + '/tokenizer')
    

    # TFBert 모델 임포트
    TF_Bert_model = TFBertModel.from_pretrained("bert-base-uncased")

    # BERT 모델 저장    
    TF_Bert_model.save_pretrained(BERT_SAVE_PATH + '/model')