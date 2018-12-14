from __future__ import print_function

import math 
import os
import tensorflow as tf
import numpy as np
import read_data
import h5py

#from PIL import Image
from six.moves import range





    
@read_data.restartable
def rgbd_dataset_generator(dataset_name, batch_size):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    
    path = '/projectnb/dl-course/MANTA' 
    file_name = dataset_name + '_s.mat' 
    f = h5py.File(os.path.join(path, file_name))
    
    X_R = f["RGB"] #shape of X now is (207920, 3, 227, 227), X_all should be (207920, 227, 227, 3),
    X_D = f["D"]
    
    y_all = f["Y"]
    y_all = np.array(y_all) #shape is (1, 207920)
    y_all = np.transpose(y_all) #shape is (207920, 1)
    
    data_len = y_all.shape[0]
    batch_size = batch_size if batch_size > 0 else data_len
    y_all[y_all == 51] = 0
    
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        
        X_all_R = X_R[idx: min(idx + batch_size,data_len)]#for padding
        X_batch_R = np.array(X_all_R) 
        X_batch_R = np.transpose(X_batch_R, (0,2,3,1)) #(batch_size, 227, 227, 3)
        
        X_batch_D = X_D[idx: min(idx + batch_size,data_len)]#for padding
        X_batch_D = np.array(X_batch_D) #(batch_size, 227, 227)
        X_batch_D = np.reshape(X_batch_D, (X_batch_D.shape[0], 227, 227,1))
       # X_batch_D = np.reshape(X_batch_D, (-1, 227, 227,1))
        
        y_batch = np.ravel(y_all[idx: min(idx + batch_size,data_len)])#for padding
        yield X_batch_R, X_batch_D, y_batch       


def mdl_rgbd(x_rbg, x_depth):
    
    #keep_prob_=0.9
    
    convR1 = tf.layers.conv2d(
            inputs=x_rbg,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="same",
            activation=tf.nn.relu)
    
    poolR1 = tf.layers.max_pooling2d(inputs=convR1, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')  # strides of the pooling operation 
    
    convR2 = tf.layers.conv2d(
            inputs = poolR1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
    poolR2 = tf.layers.max_pooling2d(inputs=convR2, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')   # strides of the pooling operation 
    
    convR3 = tf.layers.conv2d(
            inputs = poolR2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    
    convR4 = tf.layers.conv2d(
            inputs = convR3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
   
    
    convR5 = tf.layers.conv2d(
            inputs = convR4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    poolR5 = tf.layers.max_pooling2d(inputs=convR5, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')    # strides of the pooling operation 
   

    pool_flatR = tf.contrib.layers.flatten(poolR5, scope='pool2flat')
    fcR6 = tf.layers.dense(inputs=pool_flatR, units=4096, activation=tf.nn.relu)
    #fcR6Drop=tf.nn.dropout(fcR6,keep_prob_)
    fcR7 = tf.layers.dense(inputs=fcR6, units=4096, activation=tf.nn.relu)
    #fcR7Drop=tf.nn.dropout(fcR7,keep_prob_)
    
    """
     define the stram for the depth images
     
    """    
    
    convD1 = tf.layers.conv2d(
            inputs=x_depth,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="valid",
            activation=tf.nn.relu)
    
    poolD1 = tf.layers.max_pooling2d(inputs=convD1, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')  # strides of the pooling operation 
    
    convD2 = tf.layers.conv2d(
            inputs = poolD1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
    poolD2 = tf.layers.max_pooling2d(inputs=convD2, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')   # strides of the pooling operation 
    
    convD3 = tf.layers.conv2d(
            inputs = poolD2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    
    convD4 = tf.layers.conv2d(
            inputs = convD3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
   
    
    convD5 = tf.layers.conv2d(
            inputs = convD4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    poolD5 = tf.layers.max_pooling2d(inputs=convD5, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')    # strides of the pooling operation 
   
   

   
    pool_flatD = tf.contrib.layers.flatten(poolD5, scope='pool2flat')
    fcD6 = tf.layers.dense(inputs=pool_flatD, units=4096, activation=tf.nn.relu)
    #fcD6Drop=tf.nn.dropout(fcD6,keep_prob_)
    fcD7 = tf.layers.dense(inputs=fcD6, units=4096, activation=tf.nn.relu)
    #fcD7Drop=tf.nn.dropout(fcD7,keep_prob_)
   
    """
    Merging layers_fully connected
    """
    #fc8 = tf.layers.dense(inputs=tf.concat((fcR7Drop, fcD7Drop), axis=1), units=4096, activation=tf.nn.relu)
    fc8 = tf.layers.dense(inputs=tf.concat((fcR7, fcD7), axis=1), units=4096, activation=tf.nn.relu)
    fc9 = tf.layers.dense(inputs=fc8, units=51)
   
   
   
    return fc9

   
  


def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_rgb = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x_rgb')
            x_depth = tf.placeholder(tf.float32, [None, 227, 227, 1], name='x_depth')
            y_ = tf.placeholder(tf.int32, [None], name='y_')
            y_logits = model_function(x_rgb, x_depth)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)
           
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_rgb,x_depth, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict


def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                print("epoch", epoch_i, "iter",iter_i)
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('epoch {:d} iter {:d}, loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))




        
dataset_generators = {
        'train': rgbd_dataset_generator('train', 1000),
        'test': rgbd_dataset_generator('test', 1000)}
    
model_dict = apply_classification_loss(mdl_rgbd)
train_model(model_dict, dataset_generators, epoch_n=10, print_every=50)                    
