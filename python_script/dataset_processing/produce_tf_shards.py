# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 01:47:01 2022

@author: Simon
"""

import tensorflow as tf
import numpy as np
import os, sys, getopt
import pandas as pd
from random import shuffle
from multiprocessing import Pool





def getargs(argv):
    arg_csv = ""
    arg_output = ""
    arg_num_process = 3
    arg_sample_reso = 2e5
    arg_shard_size = 50
    arg_help = "{0} -d <csv_path> -o <output_path> -p <num_process> -r <num_sample_per_resolution> -s <shard_size>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hd:o:p:r:s:", ["help", "csv_path=", 
        "output_path=", "num_process=", "sample_reso=", "size_shard="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-d", "--csv_path"):
            arg_csv = arg
        elif opt in ("-o", "--output_path"):
            arg_output = arg                
        elif opt in ("-p", "--num_process"):
            arg_num_process = int(arg)                              
        elif opt in ("-r", "--sample_reso"):
            arg_sample_reso = int(arg)      
        elif opt in ("-s", "--size_shard"):
            arg_shard_size = int(arg)    

    return (arg_csv, arg_output, arg_num_process, arg_sample_reso, arg_shard_size)



def get_function_para(arg_csv, arg_output, arg_sample_reso, shard_sz):

    args = []
    file_types = {"CTU": [], "me":[], "mv":[], "trace":[]}
    
    for f in sorted(os.listdir(csv_path)):
    
        key = f.split("_")[0]
        if key in file_types.keys():
            file_types[key].append(f)
 
    assert len(file_types["CTU"]) == len(file_types["me"]) == len(file_types["mv"]) == len(file_types["trace"]), "The number of files are not correct!"
 

#Code for processing CSV files of the encoding of JVET-CTC sequences 

    # num_seq = {"240p": 0, "480p":0, "720p":0, "1080p":0, "2160p":0}  
   
    # for name_f in file_types["trace"]:
        
        # if "240p" in name_f:
            # num_seq["240p"] += 1

        # if "480p" in name_f:
            # num_seq["480p"] += 1                     

        # if "720p" in name_f:
            # num_seq["720p"] += 1    
            
        # if "1080p" in name_f:
            # num_seq["1080p"] += 1                

        # if "2160p" in name_f:
            # num_seq["2160p"] += 1                
    
    # sample_per_seq = {"240p":3*1*62, "480p":6*3*62, "720p":10*5*62, "1080p":15*8*62, "2160p":30*16*62}   



    num_seq = {"272": 0, "544":0, "720P":0, "1088":0, "2176":0} 
 

    for name_f in file_types["trace"]:
        
        if "272" in name_f:
            num_seq["272"] += 1

        if "544" in name_f:
            num_seq["544"] += 1                     

        if "720P" in name_f:
            num_seq["720P"] += 1    
            
        if "1088" in name_f:
            num_seq["1088"] += 1                

        if "2176" in name_f:
            num_seq["2176"] += 1    
            
    # for 4K sequences, we randomly select one quarter of all the encoded CTUs to store in the CSV file 
    
    sample_per_seq = {"272":3*2*62, "544":7*4*62, "720P":10*5*62, "1088":15*8*62, "2176":30*17*62 / 4}  

    num_per_res = {res: sample_per_seq[res]*num_seq[res] for res in num_seq}
    
    valid_num_s = True

    valid_seq = False

    for num in num_per_res.values():
        
        if num > 0 and num < arg_sample_reso:
            valid_num_s = False
        
        if num > 0:            
            valid_seq = True
    
    assert valid_num_s and valid_seq, "The number of samples per reso is not valid!"
    
    num_per_seq = {r: 0 if num_seq[r] == 0 else int(arg_sample_reso / num_seq[r]) for r in num_seq}
   
    for res in num_seq:
        for f in file_types["CTU"]:
            if res in f:
                args.append((arg_csv, f, arg_output, num_per_seq[res], sample_per_seq[res], shard_sz))
  
    return args
    
    
def trace_process(f, np_trace, index_list):


    ind_global = 0

    ind_sample = 0

    size_tfrecord = len(index_list)
    
    convert_table = [2, -1, 3, 1, 4, 0]
    
    bord_w = 0
    
    bord_h = 0
  

#Code for processing CSV files of the encoding of JVET-CTC sequences 

    # if "240p" in f:
        # bord_w = int(416/128)*128
        # bord_h = int(240/128)*128
    # if "480p" in f:
        # bord_w = int(832/128)*128
        # bord_h = int(480/128)*128
    # if "720p" in f:
        # bord_w = int(1280/128)*128
        # bord_h = int(720/128)*128
    # if "1080p" in f:
        # bord_w = int(1920/128)*128
        # bord_h = int(1080/128)*128
    # if "2160p" in f:
        # bord_w = int(3840/128)*128
        # bord_h = int(2160/128)*128     


        
    if "272" in f:
        bord_w = int(480/128)*128
        bord_h = int(272/128)*128
    if "544" in f:
        bord_w = int(960/128)*128
        bord_h = int(544/128)*128
    if "720P" in f:
        bord_w = int(1280/128)*128
        bord_h = int(720/128)*128
    if "1088" in f:
        bord_w = int(1920/128)*128
        bord_h = int(1088/128)*128
    if "2176" in f:
        bord_w = int(3840/128)*128
        bord_h = int(2176/128)*128
        
        

    qt_map = np.empty((size_tfrecord, 16, 16, 1), dtype=np.int8)

    qt_map.fill(-1)
    
    mt0_map = np.empty((size_tfrecord, 32, 32, 1), dtype=np.int8)

    mt0_map.fill(-1)

    mt1_map = np.empty((size_tfrecord, 32, 32, 1), dtype=np.int8)

    mt1_map.fill(-1)

    mt2_map = np.empty((size_tfrecord, 32, 32, 1), dtype=np.int8)

    mt2_map.fill(-1)


    for r in np_trace: 
        
        if r[1] >= bord_w or r[2] >= bord_h or r[0] % 32 == 0:
            continue
        
        if ind_global in index_list:
            mt_dep = int(r[12]) >> int(5*r[8])
                
            mt0_split = convert_table[mt_dep & 31]
            mt1_split = convert_table[(mt_dep >> 5) & 31]
            mt2_split = convert_table[(mt_dep >> 10) & 31]


            ref_x_mt = int((r[1] % 128) / 4)
            ref_y_mt = int((r[2] % 128) / 4)
            
            ref_x_qt = int((r[1] % 128) / 8)
            ref_y_qt = int((r[2] % 128) / 8)
            
            
            
            if (r[1] + r[3]) % 128 == 0:
                dx_qt = 16 - ref_x_qt
                dx_mt = 32 - ref_x_mt
            else:    
                dx_mt = int(((r[1] + r[3]) % 128) / 4) - ref_x_mt
                dx_qt = int(((r[1] + r[3] + 4) % 128) / 8) - ref_x_qt
                
            if (r[2] + r[4]) % 128 == 0:   
                dy_qt = 16 - ref_y_qt
                dy_mt = 32 - ref_y_mt

            else:
                dy_qt = int(((r[2] + r[4] + 4) % 128) / 8) - ref_y_qt
                dy_mt = int(((r[2] + r[4]) % 128) / 4) - ref_y_mt

            for i in range(dx_qt):
                for j in range(dy_qt):
                    assert (qt_map[ind_sample, ref_y_qt + j, ref_x_qt + i, :] == -1 or qt_map[ind_sample, ref_y_qt + j, ref_x_qt + i, :] == r[8]), "The qt size is not coherent"
                    qt_map[ind_sample, ref_y_qt + j, ref_x_qt + i, :] = r[8] 
            
            
            for i in range(dx_mt):
                for j in range(dy_mt):
                    assert (mt0_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] == -1), "The mt0 decision is not coherent"
                    assert (mt1_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] == -1), "The mt1 decision is not coherent"
                    assert (mt2_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] == -1), "The mt2 decision is not coherent"

                    mt0_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] = mt0_split
                    mt1_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] = mt1_split
                    mt2_map[ind_sample, ref_y_mt + j, ref_x_mt + i, :] = mt2_split


            if (r[1] + r[3]) % 128 == 0 and (r[2] + r[4]) % 128 == 0:
                
                if (ind_global % int(1e5) == 0):
                    print("Treating line {} with index {}".format(ind_global, ind_sample))                
                
                if ind_global in index_list:
                    ind_sample += 1
                
                ind_global += 1
        else:
            if (r[1] + r[3]) % 128 == 0 and (r[2] + r[4]) % 128 == 0:
                if (ind_global % int(1e5) == 0):
                    print("Treating line {} with index {}".format(ind_global, ind_sample))
                ind_global += 1
    
    return (qt_map, mt0_map, mt1_map, mt2_map)




def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords_file(arg_csv, f, arg_output, num_per_seq, sample_per_seq, shard_size):

    ind_sample = list(range(sample_per_seq))
    
    shuffle(ind_sample)
    
    ind_sample = ind_sample[:num_per_seq]
    
    num_tfrecord = int(num_per_seq / shard_size) + bool(num_per_seq / shard_size % 1)
    
    for i in range(num_tfrecord):
        
        end_ind = min(num_per_seq, (i+1)*shard_size) 
    
        index_list = ind_sample[i*shard_size : end_ind]
        
        size_tfrecord = end_ind - i*shard_size
        
        index_list_mv_32 = [j*5 for j in index_list] # shape 6148
        
        index_list_mv_16 = [j*5 + 1 for j in index_list] # shape 1536
        
        index_list_mv_8 = [j*5 + 2 for j in index_list] # shape 384
        
        index_list_mv_4 = [j*5 + 3 for j in index_list] # shape 96
        
        index_list_mv_2 = [j*5 + 4 for j in index_list] # shape 24
        
        
        file_name = f[4:]
        
        ctu = pd.read_csv(os.path.join(arg_csv, "CTU_" + file_name), delimiter=';', header = None, keep_default_na=False, skiprows=lambda x: x not in index_list).to_numpy()
        resi = pd.read_csv(os.path.join(arg_csv, "me_residuals_" + file_name), delimiter=';', header = None, keep_default_na=False, skiprows=lambda x: x not in index_list).to_numpy()
        
        mvf_32 = pd.read_csv(os.path.join(arg_csv, "mv_field_" + file_name), delimiter=';', header = None, usecols = list(range(6148)), keep_default_na=False, skiprows=lambda x: x not in index_list_mv_32).to_numpy()
        mvf_16 = pd.read_csv(os.path.join(arg_csv, "mv_field_" + file_name), delimiter=';', header = None, usecols = list(range(1536)), keep_default_na=False, skiprows=lambda x: x not in index_list_mv_16).to_numpy()
        mvf_8 = pd.read_csv(os.path.join(arg_csv, "mv_field_" + file_name), delimiter=';', header = None, usecols = list(range(384)), keep_default_na=False, skiprows=lambda x: x not in index_list_mv_8).to_numpy()
        mvf_4 = pd.read_csv(os.path.join(arg_csv, "mv_field_" + file_name), delimiter=';', header = None, usecols = list(range(96)), keep_default_na=False, skiprows=lambda x: x not in index_list_mv_4).to_numpy()
        mvf_2 = pd.read_csv(os.path.join(arg_csv, "mv_field_" + file_name), delimiter=';', header = None, usecols = list(range(24)), keep_default_na=False, skiprows=lambda x: x not in index_list_mv_2).to_numpy()        

        trace = pd.read_csv(os.path.join(arg_csv, "trace_RA_encoded_CU_" + file_name), delimiter=';', header = None, skiprows=67, keep_default_na=False).to_numpy()

        trace = np.reshape(trace[:, :-1], (-1, 15))    
        
        assert ctu.shape[0] == resi.shape[0] == mvf_32.shape[0], "Dimension of row is not correct."

        qptid = np.tile(np.array(resi[:, 3:5], dtype = np.float32).reshape((-1, 1, 1, 2)),  (1, 32, 32, 1))
        qptid[:, :, :, 0] = qptid[:, :, :, 0] / 46
        qptid[:, :, :, 1] = qptid[:, :, :, 1] / 5  
        
        
        ctu = np.array(ctu[:, 5:-1].reshape(-1, 128, 128, 1)/1024, dtype = np.float16)
        resi = np.array(resi[:, 5:-1].reshape(-1, 128, 128, 1)/1024, dtype = np.float16)
        
        pix = np.concatenate((ctu, resi), axis = -1)

        hor_l0_32 = np.reshape(mvf_32[:, 4:6148:6], (-1, 32, 32, 1))/2000
            
        ver_l0_32 = np.reshape(mvf_32[:, 5:6148:6], (-1, 32, 32, 1))/2000
           
        cost_l0_32 = np.reshape(mvf_32[:, 6:6148:6], (-1, 32, 32, 1))/80000


        hor_l1_32 = np.reshape(mvf_32[:, 7:6148:6], (-1, 32, 32, 1))/2000
            
        ver_l1_32 = np.reshape(mvf_32[:, 8:6148:6], (-1, 32, 32, 1))/2000
           
        cost_l1_32 = np.reshape(mvf_32[:, 9:6148:6], (-1, 32, 32, 1))/80000

        mv_32x32_input = np.concatenate((hor_l0_32, ver_l0_32, cost_l0_32, hor_l1_32, ver_l1_32, cost_l1_32), axis = -1)



        hor_l0_16 = np.reshape(mvf_16[:, 0:1536:6], (-1, 16, 16, 1))/2000
            
        ver_l0_16 = np.reshape(mvf_16[:, 1:1536:6], (-1, 16, 16, 1))/2000
           
        cost_l0_16 = np.reshape(mvf_16[:, 2:1536:6], (-1, 16, 16, 1))/80000


        hor_l1_16 = np.reshape(mvf_16[:, 3:1536:6], (-1, 16, 16, 1))/2000
            
        ver_l1_16 = np.reshape(mvf_16[:, 4:1536:6], (-1, 16, 16, 1))/2000
           
        cost_l1_16 = np.reshape(mvf_16[:, 5:1536:6], (-1, 16, 16, 1))/80000

        mv_16x16_input = np.concatenate((hor_l0_16, ver_l0_16, cost_l0_16, hor_l1_16, ver_l1_16, cost_l1_16), axis = -1)




        hor_l0_8 = np.reshape(mvf_8[:, 0:384:6], (-1, 8, 8, 1))/2000
            
        ver_l0_8 = np.reshape(mvf_8[:, 1:384:6], (-1, 8, 8, 1))/2000
           
        cost_l0_8 = np.reshape(mvf_8[:, 2:384:6], (-1, 8, 8, 1))/80000


        hor_l1_8 = np.reshape(mvf_8[:, 3:384:6], (-1, 8, 8, 1))/2000
            
        ver_l1_8 = np.reshape(mvf_8[:, 4:384:6], (-1, 8, 8, 1))/2000
           
        cost_l1_8 = np.reshape(mvf_8[:, 5:384:6], (-1, 8, 8, 1))/80000

        mv_8x8_input = np.concatenate((hor_l0_8, ver_l0_8, cost_l0_8, hor_l1_8, ver_l1_8, cost_l1_8), axis = -1)



        hor_l0_4 = np.reshape(mvf_4[:, 0:96:6], (-1, 4, 4, 1))/2000
            
        ver_l0_4 = np.reshape(mvf_4[:, 1:96:6], (-1, 4, 4, 1))/2000
           
        cost_l0_4 = np.reshape(mvf_4[:, 2:96:6], (-1, 4, 4, 1))/80000


        hor_l1_4 = np.reshape(mvf_4[:, 3:96:6], (-1, 4, 4, 1))/2000
            
        ver_l1_4 = np.reshape(mvf_4[:, 4:96:6], (-1, 4, 4, 1))/2000
           
        cost_l1_4 = np.reshape(mvf_4[:, 5:96:6], (-1, 4, 4, 1))/80000

        mv_4x4_input = np.concatenate((hor_l0_4, ver_l0_4, cost_l0_4, hor_l1_4, ver_l1_4, cost_l1_4), axis = -1)




        hor_l0_2 = np.reshape(mvf_2[:, 0:24:6], (-1, 2, 2, 1))/2000
            
        ver_l0_2 = np.reshape(mvf_2[:, 1:24:6], (-1, 2, 2, 1))/2000
           
        cost_l0_2 = np.reshape(mvf_2[:, 2:24:6], (-1, 2, 2, 1))/80000


        hor_l1_2 = np.reshape(mvf_2[:, 3:24:6], (-1, 2, 2, 1))/2000
            
        ver_l1_2 = np.reshape(mvf_2[:, 4:24:6], (-1, 2, 2, 1))/2000
           
        cost_l1_2 = np.reshape(mvf_2[:, 5:24:6], (-1, 2, 2, 1))/80000

        mv_2x2_input = np.concatenate((hor_l0_2, ver_l0_2, cost_l0_2, hor_l1_2, ver_l1_2, cost_l1_2), axis = -1)


        qtmap, mt0, mt1, mt2 = trace_process(f, trace, index_list)

        name_tfrecord = f[4:].split(".")[0] + "_" + str(i) + ".tfrecord"
        
        with tf.io.TFRecordWriter(os.path.join(arg_output, name_tfrecord)) as writer:
        
            for i in range(size_tfrecord):
                
                pix_in = tf.io.serialize_tensor(tf.convert_to_tensor(pix[i,...], dtype=np.float16))
                qpmap_in = tf.io.serialize_tensor(tf.convert_to_tensor(qptid[i,...], dtype=np.float32))
                mf32_in = tf.io.serialize_tensor(tf.convert_to_tensor(mv_32x32_input[i,...], dtype=np.float32))            
                mf16_in = tf.io.serialize_tensor(tf.convert_to_tensor(mv_16x16_input[i,...], dtype=np.float32)) 
                mf8_in = tf.io.serialize_tensor(tf.convert_to_tensor(mv_8x8_input[i,...], dtype=np.float32)) 
                mf4_in = tf.io.serialize_tensor(tf.convert_to_tensor(mv_4x4_input[i,...], dtype=np.float32)) 
                mf2_in = tf.io.serialize_tensor(tf.convert_to_tensor(mv_2x2_input[i,...], dtype=np.float32)) 
                
                
   
                qtmap_out = tf.io.serialize_tensor(tf.convert_to_tensor(qtmap[i,...], dtype=np.uint8))
      
                mt0_out = tf.io.serialize_tensor(tf.convert_to_tensor(mt0[i,...], dtype=np.uint8))
                mt1_out = tf.io.serialize_tensor(tf.convert_to_tensor(mt1[i,...], dtype=np.uint8))
                mt2_out = tf.io.serialize_tensor(tf.convert_to_tensor(mt2[i,...], dtype=np.uint8))                

                data = {'pix': _bytes_feature(pix_in.numpy()), 'qpmap': _bytes_feature(qpmap_in.numpy()), 
                        'mf2': _bytes_feature(mf2_in.numpy()), 'mf4': _bytes_feature(mf4_in.numpy()),
                        'mf8': _bytes_feature(mf8_in.numpy()), 'mf16': _bytes_feature(mf16_in.numpy()),
                        'mf32': _bytes_feature(mf32_in.numpy()), 'qtd': _bytes_feature(qtmap_out.numpy()),
                        'mt0': _bytes_feature(mt0_out.numpy()), 'mt1': _bytes_feature(mt1_out.numpy()),
                        'mt2': _bytes_feature(mt2_out.numpy())
                        }
                feature = tf.train.Features(feature=data)  
                example = tf.train.Example(features=feature)  
                serialized = example.SerializeToString()  
                writer.write(serialized)  
            
                
       


if __name__ == '__main__':
    
    
    csv_path, output_path, num_process, sam_reso, shard_sz = getargs(sys.argv)     
    args = get_function_para(csv_path, output_path, sam_reso, shard_sz)
    pool = Pool(processes = num_process)              
    pool.starmap(write_tfrecords_file, args)  






