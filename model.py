#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:02:42 2017

@author: kunhua
"""
import numpy as np
import tensorflow as tf


class seq2seq_model(object):
    
    def __init__(self, vocab_size, emb_dim, nb_layers, max_seq_len,
                 lr=0.0001, model_name='emb_att_seq2seq', ckpt_path='./checkpoint/'):
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        
        
        ## build the graph
        def __graph__():
            
            tf.reset_default_graph()
            
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            
            self.enc_input = [tf.placeholder(tf.int32, [None,]) for _ in range(max_seq_len)]
            self.targets = [tf.placeholder(tf.int32, [None,]) for _ in range(max_seq_len)]
            self.dec_input = [tf.ones_like(self.targets[0], dtype=tf.int32)] + self.targets[:-1]
    
            self.keep_prob = tf.placeholder(tf.float32)
            
            basic_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(emb_dim), 
                    output_keep_prob=self.keep_prob)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell] * nb_layers)
            
            with tf.variable_scope('decoder') as scope:
                self.dec_output_train, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.enc_input, self.dec_input, stacked_lstm, vocab_size, vocab_size, emb_dim)
                
                scope.reuse_variables()
                
                self.dec_output_infer, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.enc_input, self.dec_input, stacked_lstm, vocab_size, vocab_size, emb_dim, feed_previous=True)
            
            loss_weights = [tf.ones_like(target, dtype=tf.float32) for target in self.targets] 
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    self.dec_output_train, self.targets, loss_weights)
            
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, 
                                                                              global_step=self.global_step)
            
            
        __graph__()
        
    def get_feed(self, X, Y, keep_prob, reverse_X=True):
        if reverse_X:
            X = X[::-1]
        feed_dict = {self.enc_input[t]: X[t] for t in range(self.max_seq_len)}
        feed_dict.update({self.targets[t]: Y[t] for t in range(self.max_seq_len)})
        feed_dict[self.keep_prob] = keep_prob
        return feed_dict
    
    def train_batch(self, sess, batchX, batchY, keep_prob=1.):
        feed_dict = self.get_feed(batchX, batchY, keep_prob=keep_prob)
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
    def eval_batch(self, sess, batchX, batchY):
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        pred, loss = sess.run([self.dec_output_infer, self.loss], feed_dict)
        pred = np.array(pred).transpose([1,0,2])
        return pred, loss
    
    def evaluate(self, sess, batch_gen, nb_batches):
        losses = []
        for _ in range(nb_batches):
            batchX, batchY = batch_gen.next()
            pred, loss = self.eval_batch(sess, batchX, batchY)
            losses.append(loss)
        return np.mean(losses)
        
    def train(self, train_gen, valid_gen, steps_per_epoch, epochs, sess=None):
        
        saver = tf.train.Saver(max_to_keep=2)
        
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        
        for i in range(1, epochs+1):
            print('Epoch: {}'.format(i))
            train_losses = []
            for step in range(1, steps_per_epoch+1):
                
                try:
                    trainX, trainY = train_gen.next()
                    train_loss = self.train_batch(sess, trainX, trainY)
                    train_losses.append(train_loss)
                    if step%100 == 0:
                        print('\tstep: {}, train_loss: {:.4f}'.format(step, np.mean(train_losses)))
                        
                except KeyboardInterrupt:
                    print('Interrupted by user')
                    self.session = sess
                    return sess
                
            val_loss = self.evaluate(sess, valid_gen, valid_gen.nb_batches)
            print('val_loss: {:.4f}'.format(val_loss))
            
            saver.save(sess, 
                       self.ckpt_path + self.model_name, 
                       global_step = tf.train.global_step(sess, self.global_step))
        
        return sess
    
    def restore_last_session(self):
        saver = tf.train.Saver(max_to_keep=2)
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess
    
    def predict(self, sess, X):
        dummy_Y = np.zeros_like(X, dtype='int32')
        feed_dict = self.get_feed(X, dummy_Y, 1.)
        pred, _ = sess.run([self.dec_output_infer, self.loss], feed_dict)
        pred = np.array(pred).transpose([1,0,2])
        return pred.argmax(axis=2)
            

            
            
        