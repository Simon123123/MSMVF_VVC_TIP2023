# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 01:47:01 2022

@author: Simon
"""

import tensorflow as tf
import os, getopt, sys, math
from random import shuffle


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def getargs(argv):
    arg_shards = ""
    arg_output = ""
    arg_split = 0.0
    arg_help = "{0} -i <path_shards> -o <output_path> -r <ratio_test_split>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:r:", ["help", "path_shards=", 
        "output_path=", "ratio_test_split="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(2)
        elif opt in ("-i", "--path_shards"):
            arg_shards = arg
        elif opt in ("-o", "--output_path"):
            arg_output = arg                
        elif opt in ("-r", "--ratio_test_split"):
            arg_split = float(arg)          


    return (arg_shards, arg_output, arg_split)



if __name__ == '__main__':
    
    
    shards_path, output_path, split_ratio = getargs(sys.argv)     
    tfrecords_path = [os.path.join(shards_path, p) for p in os.listdir(shards_path)]
    shuffle(tfrecords_path)

    num_test = math.ceil(len(tfrecords_path) * split_ratio)
    
    tfrecords_path_test = tfrecords_path[:num_test]
    tfrecords_path_train = tfrecords_path[num_test:]
    
    if len(tfrecords_path_train) == 0:
        tfrecords_path_train = tfrecords_path
        tfrecords_path_test = []

    features = {'pix': tf.io.FixedLenFeature([], tf.string), 'qpmap': tf.io.FixedLenFeature([], tf.string),
                'mf2': tf.io.FixedLenFeature([], tf.string), 'mf4': tf.io.FixedLenFeature([], tf.string),
                'mf8': tf.io.FixedLenFeature([], tf.string), 'mf16': tf.io.FixedLenFeature([], tf.string),
                'mf32': tf.io.FixedLenFeature([], tf.string),'qtd': tf.io.FixedLenFeature([], tf.string),
                'mt0': tf.io.FixedLenFeature([], tf.string), 'mt1': tf.io.FixedLenFeature([], tf.string),
                'mt2': tf.io.FixedLenFeature([], tf.string)}



    with tf.io.TFRecordWriter(os.path.join(output_path, "train_final.tfrecord")) as writer:

        for tfr in tfrecords_path_train:
            
            for example in tf.data.TFRecordDataset(tfr):
          
            
                parsed_example = tf.io.parse_single_example(serialized=example, features=features) 
                                
                
                data = {'pix': _bytes_feature(parsed_example['pix'].numpy()), 'qpmap': _bytes_feature(parsed_example['qpmap'].numpy()), 
                        'mf2': _bytes_feature(parsed_example['mf2'].numpy()), 'mf4': _bytes_feature(parsed_example['mf4'].numpy()),
                        'mf8': _bytes_feature(parsed_example['mf8'].numpy()), 'mf16': _bytes_feature(parsed_example['mf16'].numpy()),
                        'mf32': _bytes_feature(parsed_example['mf32'].numpy()), 'qtd': _bytes_feature(parsed_example['qtd'].numpy()),
                        'mt0': _bytes_feature(parsed_example['mt0'].numpy()), 'mt1': _bytes_feature(parsed_example['mt1'].numpy()),
                        'mt2': _bytes_feature(parsed_example['mt2'].numpy())
                        }


                
                feature = tf.train.Features(feature=data)  
                example = tf.train.Example(features=feature)  
                serialized = example.SerializeToString()  
                writer.write(serialized)         


    
    
    if len(tfrecords_path_test) != 0:
    
        with tf.io.TFRecordWriter(os.path.join(output_path, "test_final.tfrecord")) as writer:
        
            for tfr in tfrecords_path_test:
                
                for example in tf.data.TFRecordDataset(tfr):
              
                
                    parsed_example = tf.io.parse_single_example(serialized=example, features=features) 
                                    
                    
                    data = {'pix': _bytes_feature(parsed_example['pix'].numpy()), 'qpmap': _bytes_feature(parsed_example['qpmap'].numpy()), 
                            'mf2': _bytes_feature(parsed_example['mf2'].numpy()), 'mf4': _bytes_feature(parsed_example['mf4'].numpy()),
                            'mf8': _bytes_feature(parsed_example['mf8'].numpy()), 'mf16': _bytes_feature(parsed_example['mf16'].numpy()),
                            'mf32': _bytes_feature(parsed_example['mf32'].numpy()), 'qtd': _bytes_feature(parsed_example['qtd'].numpy()),
                            'mt0': _bytes_feature(parsed_example['mt0'].numpy()), 'mt1': _bytes_feature(parsed_example['mt1'].numpy()),
                            'mt2': _bytes_feature(parsed_example['mt2'].numpy())
                            }


                    
                    feature = tf.train.Features(feature=data)  
                    example = tf.train.Example(features=feature)  
                    serialized = example.SerializeToString()  
                    writer.write(serialized)                  
                
            
    







