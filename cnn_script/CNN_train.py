# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:03:25 2021

@author: Simon
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, Dense, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D, Activation, Permute, multiply, Add, UpSampling2D, ReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import numpy as np
from tensorflow.keras.losses import MeanAbsoluteError

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from tensorflow.keras.optimizers.schedules import ExponentialDecay

from Datatset_tfrecord import get_dataset

import sys, getopt, os


config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)



def resblock(x, kernelsize, filters):
    fx = Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2D(filters, kernelsize, padding='same')(fx)
    
    x = Conv2D(filters, kernel_size=(1,1), padding='same')(x)
    
    out = Add()([x,fx])
    out = ReLU()(out)
    out = Dropout(0.2)(out)
    out = BatchNormalization()(out)
    return out



def cus_metric_qt(y_true, y_pred):
    
    return MeanAbsoluteError(y_true, y_pred)


def cnn_model():
    
    luma_size = (128, 128, 2)
    qtid_size = (32, 32, 2)


    img_input = Input(shape = luma_size)
    qpid_input = Input(shape = qtid_size)

    mf_32_input = Input(shape = (32, 32, 6))
    mf_16_input = Input(shape = (16, 16, 6))
    mf_8_input = Input(shape = (8, 8, 6))
    mf_4_input = Input(shape = (4, 4, 6))
    mf_2_input = Input(shape = (2, 2, 6))
    
    branch_1 = resblock(img_input, 5, 8)
    branch_1 = Conv2D(8, (5, 5), strides = (2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(branch_1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = Conv2D(8, (5, 5), strides = (2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(branch_1)  
    branch_1 = BatchNormalization()(branch_1)    
    
    branch_feature = Concatenate(axis = -1)([branch_1, qpid_input])

    

#   Unet branch


    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(branch_feature)
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    

#16x16

    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)    

#8x8

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  

#4x4


    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)  

#2x2

    merge5 = Concatenate(axis = -1)([pool4, mf_2_input])
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)


    up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis = -1)([drop4, up6, mf_4_input])
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis = -1)([conv3, up7, mf_8_input])
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)


    up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis = -1)([conv2, up8, mf_16_input])
    conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)


    up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = -1)([conv1, up9, mf_32_input])
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)




# branch for QTdepth map
    
    branch_qt = Conv2D(8, (3, 3), strides = (2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(conv9)

    branch_qt = BatchNormalization()(branch_qt)
    

    qt_map = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(branch_qt)
    qt_map = BatchNormalization()(qt_map)
    
    qt_map = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(qt_map)
    qt_map = BatchNormalization()(qt_map)

    qt_map = Conv2D(4, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(qt_map)
    qt_map = BatchNormalization()(qt_map)
    
    
    qt_output = Conv2D(1, (3, 3), kernel_initializer='he_uniform', padding='same', name = "qt_output")(qt_map)
  
    


#assym part for mt1 

    mt1_feature_1 = Conv2D(4, (5, 9), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt1_1")(conv9)
    mt1_feature_2 = Conv2D(8, (7, 7), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt1_2")(conv9)
    mt1_feature_3 = Conv2D(4, (9, 5), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt1_3")(conv9)

    mt1_asy_feature = Concatenate(axis = -1)([mt1_feature_1, mt1_feature_2, mt1_feature_3])

    qt_map_input = UpSampling2D((2, 2), interpolation="nearest")(qt_output)

    branch_mt1 = Concatenate(axis = -1)([qt_map_input, conv9])

    branch_mt1 = resblock(branch_mt1, 3, 16)

    branch_mt1 = Concatenate(axis = -1)([branch_mt1, mt1_asy_feature])   

    branch_mt1 = resblock(branch_mt1, 3, 16)

    branch_mt1 = resblock(branch_mt1, 3, 8)

    mt_out1 = Conv2D(5, (3, 3), kernel_initializer='he_uniform', activation='softmax', padding='same', name = "mt_out1")(branch_mt1)


    mt2_feature_1 = Conv2D(4, (3, 7), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt2_1")(conv9)
    mt2_feature_2 = Conv2D(8, (5, 5), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt2_2")(conv9)
    mt2_feature_3 = Conv2D(4, (7, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt2_3")(conv9)
 
    mt2_asy_feature = Concatenate(axis = -1)([mt2_feature_1, mt2_feature_2, mt2_feature_3])


    branch_mt2 = Concatenate(axis = -1)([qt_map_input, mt_out1, conv9])

    branch_mt2 = resblock(branch_mt2, 3, 16)

    branch_mt2 = Concatenate(axis = -1)([branch_mt2, mt2_asy_feature])   

    branch_mt2 = resblock(branch_mt2, 3, 16)   
    branch_mt2 = resblock(branch_mt2, 3, 8)

    mt_out2 = Conv2D(5, (3, 3), kernel_initializer='he_uniform', activation='softmax', padding='same', name = "mt_out2")(branch_mt2)  


    mt3_feature_1 = Conv2D(4, (1, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt3_1")(conv9)
    mt3_feature_2 = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt3_2")(conv9)
    mt3_feature_3 = Conv2D(4, (3, 1), kernel_initializer='he_uniform', activation='relu', padding='same', name = "mt3_3")(conv9)

    mt3_asy_feature = Concatenate(axis = -1)([mt3_feature_1, mt3_feature_2, mt3_feature_3])

    branch_mt3 = Concatenate(axis = -1)([qt_map_input, mt_out1, mt_out2, conv9])

    branch_mt3 = resblock(branch_mt3, 3, 16)

    branch_mt3 = Concatenate(axis = -1)([branch_mt3, mt3_asy_feature])  

    branch_mt3 = resblock(branch_mt3, 3, 16)   
    branch_mt3 = resblock(branch_mt3, 3, 8)

    mt_out3 = Conv2D(5, (3, 3), kernel_initializer='he_uniform', activation='softmax', padding='same', name = "mt_out3")(branch_mt3)       
    
    model = Model(inputs = [img_input, mf_32_input, mf_16_input, mf_8_input, mf_4_input, mf_2_input, qpid_input], outputs = [qt_output, mt_out1, mt_out2, mt_out3])



    def weighted_loss(y_true, y_pred, weight):
        
        apply_weight = K.flatten(K.sum(K.one_hot(y_true, 5) * weight, axis = -1))
        
        y_true = K.flatten(y_true)
        y_pred = K.reshape(y_pred, (-1, 5))
        return K.sum(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * tf.cast(apply_weight, tf.float32)) / tf.cast(tf.size(y_true), tf.float32)

         
    def loss_multioutput_weight(weight):

        def loss_mo_w(y_true, y_pred):
            return weighted_loss(y_true, y_pred, weight=weight)
        
        return loss_mo_w

    loss_output1 = loss_multioutput_weight(np.array([13.895, 3.934*2, 1, 4*2, 14.79]).reshape((1,1,1,5)))
    loss_output2 = loss_multioutput_weight(np.array([37.00, 12.437*2, 1, 12.315*2, 39.79]).reshape((1,1,1,5)))
    loss_output3 = loss_multioutput_weight(np.array([63, 41.28*2, 1, 44.525*2, 61.016]).reshape((1,1,1,5)))


    def accuracy_metrics(y_true, y_pred):
        acc = tf.math.confusion_matrix(labels=K.flatten(y_true), predictions=K.flatten(K.argmax(y_pred, axis=-1)))
        acc_by_true = acc/tf.reshape(K.sum(acc, axis=-1), (-1, 1))*100
        return (acc_by_true[0, 0] + acc_by_true[1,1] + acc_by_true[3, 3] + acc_by_true[4, 4])/4




    LossFunc    =     {'qt_output':'mae', 'mt_out1':loss_output1, 'mt_out2':loss_output2, 'mt_out3':loss_output3}
    lossWeights =     {'qt_output':0.8, 'mt_out1':0.2, 'mt_out2':0.2, 'mt_out3':0.2}
    outputMetrics =   {'qt_output':'mean_squared_error', 'mt_out1':['sparse_categorical_accuracy', accuracy_metrics], 'mt_out2':['sparse_categorical_accuracy', accuracy_metrics], 'mt_out3':['sparse_categorical_accuracy', accuracy_metrics]}



    lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=2000*5,
    decay_rate=0.97)
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=LossFunc, loss_weights = lossWeights, metrics=outputMetrics)
    
    print(model.summary())
    
    return model




if __name__ == '__main__':
    
    
    
    def getargs(argv):
        arg_dataset = ""
        arg_output = ""
        arg_epoch = 300
        arg_bufsize = 8000
        arg_batchsize = 400
        arg_help = "{0} -d <dataset_path> -o <output_path> -e <epoch> -bu <buffer_size> -ba <batch_size>".format(argv[0])
        
        try:
            opts, args = getopt.getopt(argv[1:], "hd:o:e:bu:ba:", ["help", "dataset_path=", 
            "output_path=", "epoch=", "buffer_size=", "batch_size="])
        except:
            print(arg_help)
            sys.exit(2)
        
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)  # print the help message
                sys.exit(2)
            elif opt in ("-d", "--dataset_path"):
                arg_dataset = arg
            elif opt in ("-o", "--output_path"):
                arg_output = arg                
            elif opt in ("-e", "--epoch"):
                arg_epoch = int(arg)                              
            elif opt in ("-bu", "--buffer_size"):
                arg_bufsize = int(arg)
            elif opt in ("-ba", "--batch_size"):
                arg_batchsize = int(arg)    
        
        return (arg_dataset, arg_output, arg_epoch, arg_bufsize, arg_batchsize)


        
    path_d, path_o, num_epoch, buf_size, bat_size = getargs(sys.argv)
    
    
    train_dataset = get_dataset(os.path.join(path_d, "train_final.tfrecord"), buf_size, bat_size)
    valid_dataset = get_dataset(os.path.join(path_d, "test_final.tfrecord"), buf_size, bat_size)
  

    model = cnn_model()
    
    
    csv_logger = CSVLogger(os.path.join(path_o, "model_history_log.csv"), append=True)
    
    checkpoint = ModelCheckpoint(os.path.join(path_o, "CNN.h5"), monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only = False, mode='min', save_freq='epoch')


    history = model.fit(train_dataset,epochs = num_epoch, validation_data = valid_dataset,
                    verbose = 2, callbacks=[csv_logger, checkpoint])    
                    



 