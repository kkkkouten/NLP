#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/02/02 19:10

@author: Tei Koten
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer  # 分词用
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 填充成相同长度
import json



if __name__=='__main__':


    # 1
    sentences = {
        'I love my dog',
        'I love my cat',
        'You love my dog!',
        'Do you think my dog is amazing?'
    }

    tokenizer = Tokenizer(num_words = 100)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences) # 创建数字数列

    print(word_index)
    print(sequences)

    # -------------------------------------------------
    test_data = [
        'i really love my dog',
        'my dog loves my manatee'
    ]
    # test_seq = tokenizer.texts_to_sequences(test_data)
    # print(test_seq)
    # -------------------------------------------------


    # 2 未知单词与填充常相同长度
    sentences = {
        'I love my dog',
        'I love my cat',
        'You love my dog!',
        'Do you think my dog is amazing?'
    }

    tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')  # 未知单词
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences) # 创建数字数列

    padded = pad_sequences(sequences,padding='post',maxlen=5,truncating='pre') # 填充成相同长度
    # padding 的方向可以改变 加入参数 padding='post'
    # maxlen = int 指定所需句子长度、
    # 如果句子长度超过了 maxlen 设置，可以设置截断方式 truncating = str
    # post 从后向前， pre 从前向后

    print(word_index)
    print(sequences)
    print(padded)

    # 3




