import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import csv 
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import math

import RBM
import time

import tensorflow.contrib.layers as layers
import sklearn.metrics as metrics

tf.set_random_seed(1000)
np.random.seed(1000)

class sigmoid_DBN(object):
    # nNet = DBN(RBM_hidden_sizes, X_train, Y_train, epochs = 80)
    def __init__(self, sizes, X, Y, eta = 0.001, momentum = 0.0, epochs = 10, batch_size = 100):
        #Initialize hyperparameters
        self._sizes = sizes
        print(self._sizes)
        self._sizes.append(10)  # [1500, 700, 400, 1000]
        self._X = X
        self._Y = Y
        self.N = len(X)   # 22939
        self.w_list = []
        self.c_list = []
        self._learning_rate = eta
        self._momentum = momentum
        self._epochs = epochs
        self._batchsize = batch_size
        input_size = X.shape[1]    #2304
        self.start_time = time.time()
        
        
        #initialization loop
        for size in self._sizes + [Y.shape[1]]:            # [1500, 700, 400, 1000, 7]
            #Define upper limit for the uniform distribution range
            max_range= math.sqrt(6./ (input_size + size))
            
            # Initialize weights through a random uniform distribution
            self.w_list.append(np.random.uniform(-max_range, max_range, [input_size,size]).astype(np.float32))
            
            #Initialize bias as zeroes
            self.c_list.append(np.zeros([size], np.float32))
            input_size = size
            
        # Build DBN
        # Create placeholders for input, weights, biases, output
        self._a = [None] * (len(self._sizes) + 2)   # [None, None, None, None, None, None]
        self._w = [None] * (len(self._sizes) + 1)   # [None, None, None, None, None]
        self._c = [None] * (len(self._sizes) + 1)   # [None, None, None, None, None]
        self._a[0] = tf.placeholder("float", [None, self._X.shape[1]])    #[None, 2304]
#         self._a[-2] = tf.placeholder("float", [None, self._X.shape[1]]) 
        self.y = tf.placeholder("float", [None, self._Y.shape[1]])    #[None, 7]
        
        # Define variables and activation function
        for i in range(len(self._sizes)+1):   # [None, None, None, None, None]
            self._w[i] = tf.Variable(self.w_list[i])
            self._c[i] = tf.Variable(self.c_list[i])
            
#         for i in range(len(self._sizes)-1):
#             self._w[i] = tf.Variable(self.w_list[i], trainable=False)
#             self._c[i] = tf.Variable(self.c_list[i], trainable=False)
        
        for i in range(1, len(self._sizes)+1 ):
            self._a[i] = tf.nn.sigmoid(tf.matmul(self._a[i - 1], self._w[i - 1]) + self._c[i - 1])
        
        self._a[3] = layers.fully_connected(self._a[2], self._sizes[-1] , activation_fn=tf.nn.sigmoid, biases_initializer=tf.zeros_initializer)
        self._a[4] = layers.fully_connected(self._a[3], self._Y.shape[1], activation_fn=None, biases_initializer=tf.zeros_initializer)
             
        # Define the cost function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self._a[-1]))
        # cost = tf.reduce)mean(tf.square(self._a[-1] - self.y))
        
        # Define the training operation (Momentum Optimizer minimizing the COst function)
        self.train_op = tf.train.AdamOptimizer(learning_rate = self._learning_rate).minimize(self.cost)
        
        # Prediction operation
        self.predict_op = tf.argmax(self._a[-1],1)
        
        # load data from rbm
    def load_from_rbms(self, dbn_sizes, rbm_list):
        # Check if expected sizes ar correct
        assert len(dbn_sizes) == len(self._sizes)   #조건에 맞지 않으면 error
            
        for i in range(len(self._sizes)):           #3
            # 각 RBM에 대한 크기가 맞는지 확인
            assert dbn_sizes[i] == self._sizes[i]
                
            # If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes) - 1):     #3    -> 0,1,2
            self.w_list[i] = rbm_list[i]._W
            self.c_list[i] = rbm_list[i]._c
#         print("self.w_list[0] : ", self.w_list[0].eval())
#         print("rbm_list[0]_W : ", rbm_list[0]._W.eval())
                
    def set_session(self, session):
        self.session = session
            
        #Training method
    def train(self, val_x, val_y):
        # For each epoch
        num_batches = self.N // self._batchsize      # 22939//100   ->  229
        train_acc_list = []
        val_acc_list = []
        time_list = []
        loss_list = []
        
        batch_size = self._batchsize
        for i in range(self._epochs):                # 80
            #For each step
            for j in range(num_batches):
                batch = self._X[j * batch_size : (j * batch_size + batch_size)]
                batch_label = self._Y[j * batch_size: (j * batch_size + batch_size)]
                    
                _, c = self.session.run([self.train_op, self.cost], feed_dict={self._a[0]: batch, self.y: batch_label})
                    
                for j in range(len(self._sizes) + 1):     #5
                        # Retrieve weights and biases
                    self.w_list[j] = self.session.run(self._w[j])
                    self.c_list[j] = self.session.run(self._c[j])
                    
                    
#                     print("w_list", self.w_list[j])
#                     print("w",self._w[i])

#             print("hi", self.w_list[0].eval())
#             print(self.w_list[1])
#             print(self.w_list[2])
#             print(self.w_list[3])
#             print(self.w_list[4])
#             print(self.w_list[5])

                        
            train_acc = np.mean(np.argmax(self._Y, axis = 1) == 
                                    self.session.run(self.predict_op, feed_dict = {self._a[0]:self._X, self.y: self._Y}))
            train_acc_list.append(train_acc)
            val_acc = np.mean(np.argmax(val_y, axis = 1) == 
                                  self.session.run(self.predict_op, feed_dict = {self._a[0]: val_x, self.y:val_y}))
            val_acc_list.append(val_acc)
            mid_time = time.time() - self.start_time
            time_list.append(mid_time)
            loss_list.append(c)

            
            print("epoch" + str(i) + "/" + str(self._epochs) + "Training Accuracy : " + str(train_acc) + " Validation Accuracy : "+str(val_acc))
            
        return train_acc_list, val_acc_list, time_list, loss_list
                
    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict = {self._a[0]: X})
                
#     def predict_ex(self,X):
#         correct_prediction = tf.equal(tf.argmax(self._a[-1],1), tf.argmax(self.y,1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         print ('Accuracy%:', accuracy.eval({self._a[0]: X_test, self.y: y_test})*100)
        
    def predict_ex(self,X,Y):
        pred_acc_list = []
        
        correct_prediction = tf.equal(tf.argmax(self._a[-1],1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_acc = accuracy.eval({self._a[0]: X, self.y: Y})*100
        
        f1_y = tf.argmax(self.y,1)
        f1_ye = f1_y.eval({self.y:Y})
        f1_p = tf.argmax(self._a[-1],1)
        f1_pe = f1_p.eval({self._a[0]:X})
        precision = metrics.precision_score(f1_ye, f1_pe)
        recall = metrics.recall_score(f1_ye, f1_pe)
        f1 = metrics.f1_score(f1_ye, f1_pe)
        fpr, tpr, _ = metrics.roc_curve(f1_ye, f1_pe)
        auc = metrics.auc(fpr, tpr)
        
        
        print ('Accuracy%:', pred_acc)
        return pred_acc, precision, recall, f1, auc
    
    
#         ri = correct_prediction.eval({self._a[0]:X, self.y:Y})*1
#         right = np.sum(ri)
#         precision = tf.argmax(self._a[-1],1)
#         precision_e = precision.eval({self._a[0]:X})
#         precision_s = right/np.sum(precision_e)
#         recall = tf.argmax(self.y,1)
#         recall_e = recall.eval({self.y:Y})
#         recall_s = right/np.sum(recall_e)
        