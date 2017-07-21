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
        """Initialize the chatbot"""
        self.root = os.getcwd()
        self.data_dir = os.path.join(self.root, 'data')
        self.name = name
        self.min_len = min_sentence_len
        self.max_len = max_sentence_len
        self.vocab_size = vocab_size
        # get a seq2seq model for the bot
        self.model = seq2seq_model(vocab_size=vocab_size, 
                                   emb_dim=512, 
                                   nb_layers=3, 
                                   max_seq_len=max_sentence_len, 
                                   model_name='emb_att_seq2seq_' + name)
        self.sess = None
        # get data for bot training 
        self._get_data()
        
    def _get_data(self):
        """Get the Cornell Movie Dialogue Corpus data """
        questions, answers, self.word2idx, self.idx2word = \
            get_data(self.data_dir, self.min_len, self.max_len, self.vocab_size-4)
        # reserve validation and test for evalution
        train, valid, test = split_data(questions, answers)
        self.data = {'train': train, 'valid': valid, 'test': test}
        
    def train(self):
        """Train the bot"""
        print('Start training...')
        train_gen = batch_generator(self.data['train'], 
                                    batch_size=128, 
                                    seq_len=self.max_len, 
                                    shuffle=True)
        valid_gen = batch_generator(self.data['valid'], 
                                    batch_size=128, 
                                    seq_len=self.max_len, 
                                    shuffle=False)
        self.sess = self.model.train(train_gen, valid_gen, train_gen.nb_batches, 
                                     epochs=100, sess=self.sess)
        
    def restore_session(self, path=None):
        """Restore certain checkpoint for the bot"""
        # use the lastest checkpoint if path is not provided
        if path is None:
            self.sess = self.model.restore_last_session()
        else:
            self.sess = self.model.restore_session(path)
    
    def evaluate(self):
        """Evaluate the bot"""
        test_gen = batch_generator(self.data['test'], 
                                    batch_size=128, 
                                    seq_len=self.max_len, 
                                    shuffle=False)
        loss = self.model.evaluate(self.sess, test_gen, test_gen.nb_batches)
        print('test_loss: {}'.format(loss))
        return loss
    
    def start_chat(self):
        """Chat with the bot"""
        # infinite loop for chatting. use Ctrl + C to stop.
        while True:
            try:
                user_input = input('user: ')
                
                # clean and tokenized user input
                question = clean_text(user_input)
                tokenized_q = tokenize(question, self.word2idx)
                
                # prepare input for the model
                x = [tokenized_q + [0]*(self.max_len - len(tokenized_q))][:self.max_len]
                x = np.array(x).transpose([1, 0])
                
                # get output from model
                y = self.model.predict(self.sess, x)[0]
                
                # translate model output to english word
                answer = detokenize(y, self.idx2word)
                print(self.name + ': ' + answer)
                
            except KeyboardInterrupt:
                break
        
                
        
        
        
        
        
        