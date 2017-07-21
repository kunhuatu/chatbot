#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:02:42 2017

@author: kunhua
"""
import numpy as np
import tensorflow as tf
import time
import os


class seq2seq_model(object):
    """The sequence-to-sequence model used for chatbot"""
    def __init__(self, vocab_size, emb_dim, nb_layers, max_seq_len,
                 lr=0.001, lr_decay=0.001, model_name='emb_att_seq2seq', ckpt_path='./checkpoint'):
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        
        def __graph__():
            """Bulid graph"""
            tf.reset_default_graph()
            
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
            self.lr_decay = tf.constant(lr_decay, dtype=tf.float32)
            new_lr = tf.maximum(self.lr * (1 / (1 + self.lr_decay*tf.cast(self.global_step, tf.float32))), 
                                tf.constant(0.00005))
            self.lr_decay_op = self.lr.assign(new_lr)
            
            # space for data feed
            self.enc_input = [tf.placeholder(tf.int32, [None,]) for _ in range(max_seq_len)]
            self.targets = [tf.placeholder(tf.int32, [None,]) for _ in range(max_seq_len)]
            self.keep_prob = tf.placeholder(tf.float32)
            
            # and add a <GO> token at the beginning of the input for decoder, and shift the rest back by one.
            self.dec_input = [tf.ones_like(self.targets[0], dtype=tf.int32)] + self.targets[:-1]
            
            # create multilayer LSTM cell the the network
            basic_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(emb_dim), 
                    output_keep_prob=self.keep_prob)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell] * nb_layers)
            
            # different situations of decoder for training and predicting
            with tf.variable_scope('decoder') as scope:
                # for training
                self.dec_output_train, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.enc_input, self.dec_input, stacked_lstm, vocab_size, vocab_size, emb_dim)
                scope.reuse_variables()
                
                # in predicting, use the output from the previous timestep as the input
                self.dec_output_infer, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.enc_input, self.dec_input, stacked_lstm, vocab_size, vocab_size, emb_dim, feed_previous=True)
            
            # define loss
            loss_weights = [tf.ones_like(target, dtype=tf.float32) for target in self.targets] 
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    self.dec_output_train, self.targets, loss_weights)
            
            # use Adam optimizer to reduce loss while training
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_decay_op).minimize(self.loss, 
                                                                                   global_step=self.global_step)
            
        # build the graph    
        __graph__()
        
    def get_feed(self, X, Y, keep_prob, reverse_X=True):
        """Get feed dictionary"""
        # reverse the input should yield better results (since bidirectional lstm is not used here)
        if reverse_X:
            X = X[::-1]
        # the dimension of X, Y should be (timestep, batch_size)
        feed_dict = {self.enc_input[t]: X[t] for t in range(self.max_seq_len)}
        feed_dict.update({self.targets[t]: Y[t] for t in range(self.max_seq_len)})
        feed_dict[self.keep_prob] = keep_prob
        return feed_dict
    
    def train_batch(self, sess, batchX, batchY, keep_prob=1.):
        """A training step for the model"""
        feed_dict = self.get_feed(batchX, batchY, keep_prob=keep_prob)
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
    def eval_batch(self, sess, batchX, batchY):
        """A evaluation step for the model"""
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        pred, loss = sess.run([self.dec_output_infer, self.loss], feed_dict)
        # transpose the dimension to (batch_size, timestep)
        pred = np.array(pred).transpose([1,0,2])
        return pred, loss
    
    def evaluate(self, sess, batch_gen, nb_batches):
        """Evaluate the model performace"""
        losses = []
        for _ in range(nb_batches):
            batchX, batchY = batch_gen.next()
            pred, loss = self.eval_batch(sess, batchX, batchY)
            losses.append(loss)
        return np.mean(losses)
        
    def train(self, train_gen, valid_gen, steps_per_epoch, epochs, sess=None):
        """Train the model"""
        # savers to save sessions
        saver_step = tf.train.Saver(max_to_keep=3, sharded=False)
        saver_epoch = tf.train.Saver(max_to_keep=5, sharded=False)
        
        # create a new session if not provided
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
        # run epochs
        for i in range(1, epochs+1):
            print('Epoch: {}'.format(i))
            train_losses = []
            
            # run steps in a epoch
            for step in range(1, steps_per_epoch+1):
                try:
                    if step == 1: start_t = time.time()
                    trainX, trainY = train_gen.next()
                    train_loss = self.train_batch(sess, trainX, trainY)
                    train_losses.append(train_loss)
                    # show log for every 100 steps
                    if step%100 == 0:
                        end_t = time.time()
                        print('\tstep: {}, train_loss: {:.4f}, time_spent: {:.2f}s'.format(step, 
                                                                                           np.mean(train_losses), 
                                                                                           end_t - start_t))
                        start_t = time.time()
                    # save session for every 200 steps
                    if step%200 == 0:
                        saver_step.save(sess, 
                                        os.path.join(self.ckpt_path, 'step', self.model_name), 
                                        global_step = tf.train.global_step(sess, self.global_step))
                # Handle the stopping by user        
                except KeyboardInterrupt:
                    print('Interrupted by user')
                    return sess
                
            # evaluated by validation data at the end of each epoch
            val_loss = self.evaluate(sess, valid_gen, valid_gen.nb_batches)
            print('val_loss: {:.4f}'.format(val_loss))
            
            # save session for every 5 epochs
            if i%5 == 0:
                saver_epoch.save(sess, 
                                 os.path.join(self.ckpt_path, 'epoch', self.model_name), 
                                 global_step = tf.train.global_step(sess, self.global_step))
        return sess
    
    def restore_last_session(self):
        """Get the last session"""
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path + '/step/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return sess
    
    def restore_session(self, path):
        """Get the designated session"""
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, path)
        return sess

    def predict(self, sess, X):
        """Predict the answering sentence"""
        # create dummy to satisfy the input, but won't be used in predicting
        dummy_Y = np.zeros_like(X, dtype='int32')
        feed_dict = self.get_feed(X, dummy_Y, 1.)
        pred, _ = sess.run([self.dec_output_infer, self.loss], feed_dict)
        # transpose the dimension to (batch_size, timestep)
        pred = np.array(pred).transpose([1,0,2])
        return pred.argmax(axis=2)
            

            
            
        