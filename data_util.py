#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 06:58:57 2017

@author: kunhua
"""

import pandas as pd
import numpy as np
import math
import re
import os
import time

def get_id2line(fname):
    lines = open(fname, encoding='utf-8', errors='ignore').read().split('\n')
    id2line = dict()
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = clean_text(_line[4])
    return id2line


def get_conversations(fname):
    conv_lines = open(fname, encoding='utf-8', errors='ignore').read().split('\n')
    convs = []
    for line in conv_lines:
        _line = line.split(' +++$+++ ')[-1]
        convs.append(re.findall("L[0-9]+", _line))
    return convs


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) 
    return text


def get_QA(convs, id2line):
    questions, answers = [], []
    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i+1]])
    return questions, answers


def filter_QA(questions, answers, min_len, max_len):
    filtered_q, filtered_a = [], []
    assert len(questions) == len(answers)
    for (q, a) in zip(questions, answers):
        if min_len <= len(q.split()) <= max_len and min_len <= len(a.split()) <= max_len:
            filtered_q.append(q)
            filtered_a.append(a)
    return filtered_q, filtered_a


def get_index_dict(sentences, vocab_size):
    freq_dict = dict()
    for sentence in sentences:
        for word in sentence.split():
            freq_dict[word] = freq_dict.get(word, 0) + 1
    vocab = sorted(freq_dict.items(), key=lambda x: -x[1])[:vocab_size]
    idx2word = ['<PAD>', '<GO>', '<UNK>', '<EOS>'] + [x[0] for x in vocab]
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])
    return idx2word, word2idx


def tokenize_QA(questions, answers, word2idx):
    tokenized_q, tokenized_a = [], []
    for (q, a) in zip(questions, answers):
        q, a = q.split(), a.split()
        unk_count_q = sum([w not in word2idx for w in q])
        unk_count_a = sum([w not in word2idx for w in a])
        if unk_count_a > 2 or (unk_count_q/len(q)) > 0.2:
            continue
        unk_idx = word2idx['<UNK>']
        tokenized_q.append([word2idx.get(w, unk_idx) for w in q])
        tokenized_a.append([word2idx.get(w, unk_idx) for w in a] + [word2idx['<EOS>']])
    return tokenized_q, tokenized_a


def tokenize(sentence, word2idx):
    unk_idx = word2idx['<UNK>']
    sentence = sentence.split()
    return [word2idx.get(w, unk_idx) for w in sentence]


def detokenize(sentence, idx2word):
    sentence = [idx2word[idx] for idx in sentence]
    try:
        end_pos = sentence.index('<EOS>')
    except ValueError:
        end_pos = len(sentence)
        
    clean_sentence = []
    for word in sentence[:end_pos]:
        if word not in ['<PAD>', '<GO>', '<UNK>', '<EOS>']:
            clean_sentence.append(word)
            
    return ' '.join(clean_sentence)
    

def get_data(data_folder, min_len, max_len, vocab_size):
    id2line = get_id2line(os.path.join(data_folder, 'movie_lines.txt'))
    convs = get_conversations(os.path.join(data_folder, 'movie_conversations.txt'))
    questions, answers = get_QA(convs, id2line)
    f_questions, f_answers = filter_QA(questions, answers, min_len, max_len-1)
    idx2word, word2idx = get_index_dict(id2line.values(), vocab_size)
    tokenized_q, tokenized_a = tokenize_QA(f_questions, f_answers, word2idx)
    return tokenized_q, tokenized_a, word2idx, idx2word


def split_data(x, y, ratio=[0.7, 0.15, 0.15]):
    ratio = np.array(ratio) / sum(ratio)
    data_len = len(x)
    lengths = [int(data_len * r) for r in ratio]
    trainX, trainY = x[:lengths[0]], y[:lengths[0]]
    validX, validY = x[lengths[0] : lengths[0]+lengths[1]], y[lengths[0] : lengths[0]+lengths[1]]
    testX, testY = x[-lengths[-1]:], y[-lengths[-1]:]
    return (trainX, trainY), (validX, validY), (testX, testY)




class batch_generator(object):
    
    def __init__(self, data, batch_size, seq_len, shuffle=False):
        self.data = np.array(data)
        self.N = len(data[0])
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nb_batches = int(math.ceil(self.N / batch_size))
        self.idx_array = np.arange(self.N)
        self.idx_pos = 0
        self.shuffle = shuffle
        
    def reset(self):
        self.idx_pos = 0
        self.idx_array = np.arange(self.N)
        
    def _get_batch_indices(self):
        if self.idx_pos >= self.N:
            self.reset()
        if self.idx_pos == 0 and self.shuffle:
            np.random.shuffle(self.idx_array)
        batch_indices = self.idx_array[self.idx_pos : self.idx_pos + self.batch_size]
        self.idx_pos += self.batch_size
        return batch_indices
    
    def _get_batch_data(self, batch_indices):
        X = self.data[0][batch_indices]
        Y = self.data[1][batch_indices]
        X = self._pad_data(X, self.seq_len)
        Y = self._pad_data(Y, self.seq_len)
        return X.transpose([1, 0]), Y.transpose([1, 0])
        
    def _pad_data(self, data, seq_len):
        padded_data = [sentence + [0] * (seq_len - len(sentence)) for sentence in data]
        return np.array(padded_data, dtype='int32')
    
    def __next__(self):
        batch_indices = self._get_batch_indices()
        return self._get_batch_data(batch_indices)
    
    def next(self):
        return self.__next__()
    
    
        
        
    


