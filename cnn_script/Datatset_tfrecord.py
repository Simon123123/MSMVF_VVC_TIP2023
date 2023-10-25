# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 01:47:01 2022

@author: Simon
"""

import tensorflow as tf
import keras.backend as K

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_example(serialized):
    features = {'pix': tf.io.FixedLenFeature([], tf.string), 'qpmap': tf.io.FixedLenFeature([], tf.string),
                'mf2': tf.io.FixedLenFeature([], tf.string), 'mf4': tf.io.FixedLenFeature([], tf.string),
                'mf8': tf.io.FixedLenFeature([], tf.string), 'mf16': tf.io.FixedLenFeature([], tf.string),
                'mf32': tf.io.FixedLenFeature([], tf.string),'qtd': tf.io.FixedLenFeature([], tf.string),
                'mt0': tf.io.FixedLenFeature([], tf.string), 'mt1': tf.io.FixedLenFeature([], tf.string),
                'mt2': tf.io.FixedLenFeature([], tf.string)}
                
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)


    pixel = parsed_example['pix'] 
    pixel = tf.io.parse_tensor(pixel, tf.float16)    
    pixel = tf.reshape(pixel, [128, 128, 2])
    


    qp = parsed_example['qpmap']  
    qp = tf.io.parse_tensor(qp, tf.float32)     
    qp = tf.reshape(qp, [32, 32, 2])



    mf2x2 = parsed_example['mf2'] 
    mf2x2 = tf.io.parse_tensor(mf2x2, tf.float32)       
    mf2x2 = tf.reshape(mf2x2, [2, 2, 6])

    mf4x4 = parsed_example['mf4']  
    mf4x4 = tf.io.parse_tensor(mf4x4, tf.float32)      
    mf4x4 = tf.reshape(mf4x4, [4, 4, 6])

    mf8x8 = parsed_example['mf8']  
    mf8x8 = tf.io.parse_tensor(mf8x8, tf.float32)      
    mf8x8 = tf.reshape(mf8x8, [8, 8, 6]) 


    mf16x16 = parsed_example['mf16']  
    mf16x16 = tf.io.parse_tensor(mf16x16, tf.float32)     
    mf16x16 = tf.reshape(mf16x16, [16, 16, 6])


    mf32x32 = parsed_example['mf32']  
    mf32x32 = tf.io.parse_tensor(mf32x32, tf.float32)       
    mf32x32 = tf.reshape(mf32x32, [32, 32, 6])


    qtdepth = parsed_example['qtd']  
    qtdepth = tf.io.parse_tensor(qtdepth, tf.uint8)        
    qtdepth = tf.reshape(qtdepth, [16, 16, 1])
    
    mtd0 = parsed_example['mt0']  
    mtd0 = tf.io.parse_tensor(mtd0, tf.uint8)      
    mtd0 = tf.reshape(mtd0, [32, 32, 1])
    
    mtd1 = parsed_example['mt1']  
    mtd1 = tf.io.parse_tensor(mtd1, tf.uint8)       
    mtd1 = tf.reshape(mtd1, [32, 32, 1])
    
    mtd2 = parsed_example['mt2'] 
    mtd2 = tf.io.parse_tensor(mtd2, tf.uint8)        
    mtd2 = tf.reshape(mtd2, [32, 32, 1]) 


    X = (pixel,  mf32x32, mf16x16, mf8x8, mf4x4, mf2x2, qp)
#    
    Y = (qtdepth, mtd0, mtd1, mtd2)


    return X, Y 





def get_dataset(data_path, buf_size, batch_size):

    train_dataset = tf.data.TFRecordDataset(data_path)  
    train_dataset = train_dataset.map(parse_example)  
    train_dataset = train_dataset.shuffle(int(buf_size)).batch(int(batch_size))
    return train_dataset








