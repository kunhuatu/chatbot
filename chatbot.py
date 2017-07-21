#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:30:06 2017

@author: kunhua
"""
from model import seq2seq_model
import data_util
from data_util import get_data, split_data, batch_generator
from data_util import clean_text, tokenize, detokenize
import numpy as np
import os


class ChatBot(object):
    
    def __init__(self, name, min_sentence_len=2, max_sentence_len=20, vocab_size=8000):
        self.root = os.getcwd()
        self.data_dir = os.path.join(self.root, 'data')
        self.name = name
        self.min_len = min_sentence_len
        self.max_len = max_sentence_len
        self.vocab_size = vocab_size
        
        self.model = seq2seq_model(vocab_size=vocab_size, 
                                   emb_dim=256, 
                                   nb_layers=3, 
                                   max_seq_len=max_sentence_len, 
                                   model_name='emb_att_seq2seq' + name)
        self.sess = None
        
        self._get_data()
        
    def _get_data(self):
        questions, answers, self.word2idx, self.idx2word = \
            get_data(self.data_dir, self.min_len, self.max_len, self.vocab_size)
        train, valid, test = split_data(questions, answers)
        self.data = {'train': train, 'valid': valid, 'test': test}
        
    def train(self):
        
        print('Start training...')
        train_gen = batch_generator(self.data['train'], 
                                    batch_size=64, 
                                    seq_len=self.max_len, 
                                    shuffle=True)
        valid_gen = batch_generator(self.data['valid'], 
                                    batch_size=128, 
                                    seq_len=self.max_len, 
                                    shuffle=False)
        
        self.sess = self.model.train(train_gen, valid_gen, train_gen.nb_batches, 
                                     epochs=10, sess=self.sess)
        
    def restore_session(self):
        self.sess = self.model.restore_last_session()
    
    def evaluate(self):
        test_gen = batch_generator(self.data['test'], 
                                    batch_size=128, 
                                    seq_len=self.max_len, 
                                    shuffle=False)
        loss = self.model.evaluate(self.sess, test_gen, test_gen.nb_batches)
        return loss
    
    def start_chat(self):
        
        while True:
            try:
                user_input = input(': ')
                question = clean_text(user_input)
                tokenized_q = tokenize(question, self.word2idx)
                x = [tokenized_q + [0]*(self.max_len - len(tokenized_q))][:self.max_len]
                x = np.array(x).transpose([1, 0])
                y = self.model.predict(self.sess, x)[0]
                answer = detokenize(y, self.idx2word)
                print(answer)
                
            except KeyboardInterrupt:
                break
        
                
        
        
        
        
        
        