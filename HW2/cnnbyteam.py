# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 01:22:21 2019

@author: jason
"""

import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#################################################################################################
Start = time.time()
######################
mnist = input_data.read_data_sets ( "MNIST_data/" , one_hot=True )
training_data = mnist.train.images
training_label = mnist.train.labels
testing_data = mnist.test.images
testing_label = mnist.test.labels
##########################################################################################################################################
##########################################################################################################################################
i=1
#print(training_data[i])
z = training_data[i].reshape(28,28)
#im=Image.fromarray(y) # numpy 转 image类
#im.show()
#################################################################################################
pooling=2
filter_num1=3
filter_w1=3
filter_h1=3
##########################
filter_num2=3
filter_w2=3
filter_h2=3
##########################
learning_rate = 2/10**15
Training_epochs = 100
Training_time = 100
train=55000
test=10000
#################################################################################################
filter1 = np.random.randint(1,10,[filter_num1,filter_h1,filter_w1])/100
filter2 = np.random.randint(1,10,[filter_num2,filter_h2,filter_w2])/100
##########################################################################################################################################
def feature_map(image_source,filter):
#    image_source = y
#    filter = filter1
    dim=image_source.shape #(height, width, channel)
    F = filter.shape
    f1,f2,f3 = F
    if len(dim) != 2:
        C, H, W = dim
        feature_img = np.zeros((f1*C,H+3-f3,W+3-f2),dtype=float)
        new_img = np.zeros((C,H+2,W+2),dtype=float)
        new_img[0:C,1:H+1,1:W+1] = image_source
        H_numberli = list(range(H+3-f3))
        W_numberli = list(range(W+3-f2))
        C_numberli = list(range(C))
        F_numberli = list(range(f1))
        f = 0 ; c = 0 ; g = 0
        for f_index in F_numberli:
            for c_index in C_numberli:
                if f != f_index or c != c_index :
                            g = g + 1
                            f=f_index;c=c_index
                for h_index in H_numberli:
                    for w_index in W_numberli:
                        feature_img[g,h_index,w_index]= sum(sum(filter[f_index].dot(new_img[c_index,h_index:h_index+f3,w_index:w_index+f2])))
    else:
        H, W = dim
        feature_img = np.zeros((f1,H+3-f3,W+3-f2),dtype=float)
        new_img = np.zeros((H+2,W+2),dtype=float)
        new_img[1:H+1,1:W+1] = image_source
        H_numberli = list(range(H+3-f3))
        W_numberli = list(range(W+3-f2))
        F_numberli = list(range(f1))
        for h_index in H_numberli:
            for w_index in W_numberli:
                for f_index in F_numberli:
                    feature_img[f_index,h_index,w_index]= sum(sum(filter[f_index].dot(new_img[h_index:h_index+f3,w_index:w_index+f2])))
    return  feature_img
#################################################
#map=feature_map(y,filter1)
#im=Image.fromarray(map[0]*255) # numpy 转 image类
#im.show()
########################################################################################################################################
def reLu(reLu_map):
#    reLu_map=map
    R=reLu_map.shape
    r1,r2,r3 = R
    map_reLu = np.zeros((r1,r2,r3),dtype=float)
    map_reLu = np.maximum(reLu_map, 0)
    return map_reLu
#################################################
#re=reLu(map)
#im=Image.fromarray(re[0]*255) # numpy 转 image类
#im.show()
#######################################################################################################################################
def max_pooling(im,pooling):
#    im=re
    I=im.shape #(height, width, channel)
    i1,i2,i3 = I
    pooling_img = np.zeros((i1,i3//pooling,i2//pooling),dtype=float)
    H_numberli = list(range(i3//pooling))
    W_numberli = list(range(i2//pooling))
    M_numberli = list(range(i1))
    for m_index in M_numberli:
        for h_index in H_numberli:
            for w_index in W_numberli:
                pooling_img[m_index,h_index,w_index]= np.max(im[m_index,
                           h_index*pooling:h_index*pooling+pooling,w_index*pooling:w_index*pooling+pooling])
    return  pooling_img
#################################################
#pl=max_pooling(re,pooling)
#im=Image.fromarray(pl[0]*255) # numpy 转 image类
#im.show()
######################################################################################################################################
def Convolution():
    step1_fm = feature_map(z,filter1)
    step1_relu = reLu(step1_fm)
    step1_pool = max_pooling(step1_relu,pooling)
    step2_fm = feature_map(step1_pool,filter2)
    step2_relu = reLu(step2_fm)
    step2_pool = max_pooling(step2_relu,pooling)
    return step2_pool
###################################################
final=Convolution()
#im=Image.fromarray(final[0]*255) # numpy 转 image类
#im.show()
f1,f2,f3=final.shape
#z = final.reshape(1,f1*f2*f3)
######################################################################################################################################
#fully connected layers
######################################################################################################################################
#neurons_number
input_number=f1*f2*f3
neurons_1_number=200
neurons_2_number=100
neurons_3_number=100
neurons_4_number=10

train_loss_col = []
test_loss_col = []
train_layer_4 = []
train_y = []
test_y = []
test_layer_4 = []
###########################################################################################
w1=np.random.randint(1,10,[input_number,neurons_1_number])/1000 #(1,10）以內的X行X列隨機整數
w2=np.random.randint(1,10,[neurons_1_number,neurons_2_number])/1000
w3=np.random.randint(1,10,[neurons_2_number,neurons_3_number])/1000
w4=np.random.randint(1,10,[neurons_3_number,neurons_4_number])/1000
###########################################################################################
b1=np.random.randint(1,10,[neurons_1_number,1])/100 #(1,10）以內的X行X列隨機整數
b2=np.random.randint(1,10,[neurons_2_number,1])/100
b3=np.random.randint(1,10,[neurons_3_number,1])/100
b4=np.random.randint(1,10,[neurons_4_number,1])/100
###########################################################################################
##Training
###########################################################################################
for t in range(Training_time):
#############################################
    for i in range(0,train):
#############################################
        z = training_data[i].reshape(28,28)
        final=Convolution()
#############################################
        inp=final.reshape(1,input_number).T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp.T,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=training_label[i]

        # Compute and print loss
        train_loss = np.square(layer_4 - y).sum() # loss function
        train_loss_col.append(train_loss)
        train_y.append(y)
        train_layer_4.append(layer_4)
      ###################################################################
        # Backprop to compute gradients of weights with respect to loss
        grad_layer_4 = 2.0 * (layer_4 - y) # the last layer's error
        a=np.ones((neurons_3_number,1))*grad_layer_4
        grad_w4 = layer_3.T.dot(a)
        grad_layer_3 = grad_layer_4.dot(w4.T) # the second laye's error
        grad_h3 = grad_layer_3.copy()
        grad_h3[layer_3 < 0] = 0  # the derivate of ReLU
        b=np.ones((neurons_2_number,1))*grad_h3
        grad_w3 = layer_2.T.dot(b)
        grad_layer_2 = grad_layer_3.dot(w3.T) # the second laye's error
        grad_h2 = grad_layer_2.copy()
        grad_h2[layer_2 < 0] = 0  # the derivate of ReLU
        c=np.ones((neurons_1_number,1))*grad_h2
        grad_w2 = layer_1.T.dot(c)
        grad_layer_1 = grad_layer_2.dot(w2.T) # the second laye's error
        grad_h1 = grad_layer_1.copy()
        grad_h1[layer_1 < 0] = 0  # the derivate of ReLU
        d=np.ones((input_number,1))*grad_h1
        grad_w1 = inp.T.dot(d)
        # Update weights
        w1 = w1-(learning_rate * w1*(grad_w1))
        w2 = w2-(learning_rate * w2*(grad_w2))
        w3 = w3-(learning_rate * w3*(grad_w3))
        w4 = w4-(learning_rate * w4*(grad_w4))
     ##################################################################
        # Backprop to compute gradients of bias with respect to loss
        grad_b4 = layer_3.T.dot(a)
        grad_b3 = layer_2.T.dot(b)
        grad_b2 = layer_1.T.dot(c)
        grad_b1 = inp.T.dot(d)
        # Update bias
        b1 = b1-(learning_rate * b1*(sum(grad_b1)))
        b2 = b2-(learning_rate * b2*(sum(grad_b2)))
        b3 = b3-(learning_rate * b3*(sum(grad_b3)))
        b4 = b4-(learning_rate * b4*(sum(grad_b4)))
#    if train_loss <= 0.001 : break
########################################################################
plt.plot(train_loss_col)
plt.show()
plt.figure ( figsize = ( 20 , 10 ))
plt.plot ( train_layer_4 , label = "$output$" , color = "red" , linewidth = 1 )
plt.plot ( train_y , "b--" , label = "$label$" , linewidth = 0.3)
plt.xlabel ( "Time(s)" )
plt.ylabel( "Value" )
plt.title ( "Training" )
plt.legend ()
plt.show ()
############################################################################################
############################################################################################
##Testing
############################################################################################
for i in range(0,test):
#    for i in range(0, 50):
###############################################
        z = testing_data[i].reshape(28,28)
        final=Convolution()
###############################################
        inp=final.reshape(1,input_number).T
        # Forward pass: compute predicted y
        r_1=sum(np.dot(inp.T,w1)+b1)
        layer_1=np.maximum(r_1, 0)
        r_2=sum(np.dot(layer_1,w2)+b2)
        layer_2=np.maximum(r_2, 0)
        r_3=sum(np.dot(layer_2,w3)+b3)
        layer_3=np.maximum(r_3, 0)
        r_4=sum(np.dot(layer_3,w4)+b4)
        layer_4=np.maximum(r_4, 0)
        y=testing_label[i]
        # Compute and print loss
        test_loss = np.square(layer_4 - y).sum() # loss function
        test_loss_col.append(test_loss)
        test_y.append(y)
        test_layer_4.append(layer_4)
plt.plot(test_loss_col)
plt.show()
###############################################################################
plt.figure ( figsize = ( 20 , 10 ))
plt.plot ( test_layer_4 , label = "$output$" , color = "red" , linewidth = 2 )
plt.plot ( test_y , "b" , label = "$label$" , linewidth = 1)
plt.xlabel ( "Time(s)" )
plt.ylabel( "Value" )
plt.title ( "Testing" )
plt.legend ()
###########################################################################################
End = time.time()
print("Total %f sec" % (End - Start))
