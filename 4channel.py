from __future__ import print_function

import math 
import os
import tensorflow as tf
import numpy as np
import read_data
import h5py
#from PIL import Image
from six.moves import range
from skimage import img_as_ubyte
import warnings

#Read dataset
@read_data.restartable
def rgbd_dataset_generator(dataset_name, batch_size):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    
    path = '/projectnb/dl-course/MANTA' 
    file_name = dataset_name + '_normal_4ch.mat' 
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_batch_D = img_as_ubyte(X_batch_D) 
       
        X_batch = np.concatenate( (X_batch_R, X_batch_D), axis=3)
        
        y_batch = np.ravel(y_all[idx: min(idx + batch_size,data_len)])#for padding
        yield X_batch, y_batch     

# 4 Channel RGBD Model
def mdl_rgbd(x_):
    
    kernel = []
    bias = []
    
    with tf.variable_scope("Hidden1RGB"):
    
        convR1 = tf.layers.conv2d(
            inputs=x_,
            filters= 10,  # number of filters
            strides= 4, # convolution stride
            kernel_size=[9, 9],
            padding="same",
            activation=tf.nn.relu,name="convR1")
    
        poolR1 = tf.layers.max_pooling2d(inputs=convR1, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid',name="poolR1")  # strides of the pooling operation 
                                    
        tf.get_variable_scope().reuse_variables()                            
        kernel.append(tf.get_variable('convR1/kernel'))
        bias.append(tf.get_variable('convR1/bias'))
        
    with tf.variable_scope("Hidden2RGB"):
    
        convR2 = tf.layers.conv2d(
            inputs = poolR1,
            filters = 10,
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu,name="convR2")
    
        poolR2 = tf.layers.max_pooling2d(inputs=convR2, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid',name="poolR2")   

        tf.get_variable_scope().reuse_variables()                            
        kernel.append(tf.get_variable('convR2/kernel'))
        bias.append(tf.get_variable('convR2/bias'))
        
    with tf.variable_scope("Hidden3RGB"):
    
        convR3 = tf.layers.conv2d(
            inputs = poolR2,
            filters = 10, 
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu,name="convR3")
    
        poolR3 = tf.layers.max_pooling2d(inputs=convR3, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid',name="poolR3")    
   
		
        tf.get_variable_scope().reuse_variables()                            
        kernel.append(tf.get_variable('convR3/kernel'))
        bias.append(tf.get_variable('convR3/bias'))   

    with tf.variable_scope("Hidden4RGB"):  

        pool_flatR = tf.contrib.layers.flatten(poolR3, scope='poolflat')
        fcR4 = tf.layers.dense(inputs=pool_flatR, units=64, activation=tf.nn.relu,name="fcR4")
        
        tf.get_variable_scope().reuse_variables()                            
        kernel.append(tf.get_variable('fcR4/kernel'))
        bias.append(tf.get_variable('fcR4/bias'))    
  

    with tf.variable_scope("FullyConnected"):
        fc5 = tf.layers.dense(inputs=fcR4, units=51,name="fc5")
        tf.get_variable_scope().reuse_variables()    
    	                        
        kernel.append(tf.get_variable('fc5/kernel'))
        bias.append(tf.get_variable('fc5/bias'))

    return [fc5,kernel,bias,convR1,convR3,fcR4]

def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 227, 227, 4], name='x_')
            y_ = tf.placeholder(tf.int32, [None], name='y_')
            y_logits,kernels,biases,convR1,convR3,fcR4 = model_function(x_)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)
           
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        with tf.device("/cpu:0"):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
            image_shaped_input = tf.reshape(kernels[0], [-1, 9, 9, 4]) # RGB [-1, 5, 5, 3]:Grey
            tf.summary.image('kernel', image_shaped_input, 227)
            tf.summary.image('convR1/activation', tf.reshape(convR1[0],[-1,57,57,1]))
            merge = tf.summary.merge_all()
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss, 'merge':merge,'kernel':kernels, 'bias':biases, 
                  'convR1':convR1, 'convR3':convR3, 'fc4':fcR4}
    
    return model_dict

def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_writer = tf.summary.FileWriter('Summary/',graph=model_dict['graph'])
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):

                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                -
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                        summary = sess.run(model_dict['merge'], feed_dict = test_feed_dict)
                        test_writer.add_summary(summary, epoch_i*140+iter_i)
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('epoch {:d} iter {:d}, loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))

dataset_generators = {
        'train': rgbd_dataset_generator('train', 100),
        'test': rgbd_dataset_generator('test', 100)}

model_dict = apply_classification_loss(mdl_rgbd)
train_model(model_dict, dataset_generators, epoch_n=30, print_every=50)