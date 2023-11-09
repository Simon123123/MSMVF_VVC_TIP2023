CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields
============================================================

This is the source code for the paper **CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields** (cf. https://arxiv.org/abs/2310.13838) 
currently under review of IEEE Transactions on Image Processing.

Our dataset MVF-Inter is available at https://1drv.ms/f/s!Aoi4nbmFu71Hgx9FJphdskXfgIVo?e=fXrs0o


For reusing the code in this project, please think about citing this paper. Thanks! If you have further questions, please contact me at liusimon0914@gmail.com.

Build instructions
------------------

**It is generally suggested to build 64-bit binaries for VTM software**. We have built the software on Windows (MS Visual Studio) and Linux (make).  



**Windows Visual Studio 64 Bit:**

Use the proper generator string for generating Visual Studio files, e.g. for VS 2019:

```bash
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

Then open the generated solution file in MS Visual Studio.

For Visual Studio 2017, use "Visual Studio 15 2017 Win64", for Visual Studio 2019 use "Visual Studio 16 2019".

Visual Studio 2019 also allows you to open the CMake directory directly. Choose "File->Open->CMake" for this option.

For the release build of this project in Visual Studio, we should set the EncoderApp as Startup Project. Before building it, 
right-click on the EncoderLib and set "Treat Warnings As Error" to "No" in "Property->C/C++/General". Then add the compile option 
/bigobj in "Property->C/C++/Command Line" for EncoderLib. 
 

**Linux**

For generating Linux Release Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
For generating Linux Debug Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Then type
```bash
make -j
```

For more details, refer to the CMake documentation: https://cmake.org/cmake/help/latest/

Dataset generation and processing
---------------------------------

To create the dataset, activate the macros **MSMVF_GLOBAL** and **MSMVF_DATASET** defined respectively at line 56 and 60 in the file **TypeDef.h**. If you want to generate a dataset 
containing reference pixels as an input feature (cf. The training of **PIX-CNN** in our paper), please activate the **MSMVF_REFPIX** macro. The **MSMVF_4k** macro needs to be activated to generate data for 4K sequences. 
As described in our paper, we encode the first 64 frames for sequences in [BVI-DVI dataset](https://fan-aaron-zhang.github.io/BVI-DVC/) and YouTube UGC dataset (available at https://1drv.ms/f/s!Aoi4nbmFu71Hg2Gj4LW-eDUMVVo0?e=r3htiz) at QP 22, 27, 32 and 37. After encoding the sequence with the EncoderApp, 
four (five if **MSMVF_REFPIX** is enabled) CSV files are generated: **trace_RA_encoded_CU_seq_name.csv**, **me_residuals_seq_name.csv**, **CTU_seq_name.csv**, and **mv_field_seq_name.csv** (plus **RefPixel_seq_name.csv**). 


To process these generated CSV files and create a large TFRecord dataset, follow these steps:


1. Put the CSV files in a folder. 

2. Execute the **produce_tf_shards.py** script located in **python_script\dataset_processing** to generate small shards of TFRecord files.  If you want to randomly select a certain number of samples per video resolution and store these selected samples in numerous shards, 
run the following command: 


```
python produce_tf_shards.py -d <csv_path> -o <output_path> -p <num_process> -r <num_sample_per_resolution> -s <num_sample_per_shard>
```

This script operates in a multi-process manner, and each encoding corresponds to one process. **num_process** represents the maximum number of parallel processes.


3.  Next, execute the **merge_shards.py** script located in **python_script\dataset_processing** to combine these shards into two large TFRecord files, one for the training set and the other for the test set:

```
python merge_shards.py -i <path_shards> -o <output_path> -r <ratio_test_split>
```


It's important to note that **ratio_test_split** should be a value between 0 and 1, indicating the proportion of samples in the test set. For instance, ratio_test_split = 0.2 signifies that 20\% of the samples are in the test set. 
The **output_path** should only contain the path without the dataset file name. After running this script, two TFRecord files, named **train_final.tfrecord** and **test_final.tfrecord**, will be created in the **output_path**.



With regard to managing large TFRecord datasets, there are two common approaches. One involves storing samples in small shards, and the other involves storing samples in a single, large TFRecord file. Each method has its advantages and disadvantages. 
We have chosen the latter for our TF dataset. For more details about the TFRecord file format (e.g., example protocol, etc.), please refer to the Python scripts.



Training instructions
---------------------


To train the related CNN network as described in the paper, please download the aforementioned TFRecord dataset. Before starting the training, ensure that the following packages are installed in your training environment: **Keras**, **numpy**, and **TensorFlow GPU**.

You can then use the scripts in the **python_script\cnn_training** folder to train the CNN. The Python script **CNN_train.py** is used to define the CNN structure and initiate its training.

To run this script, use the following command:


```
python CNN_train.py -d <dataset_path> -o <output_path> -e <epoch> -bu <buffer_size> -ba <batch_size>
```

Here, **buffer_size** indicates the size of the buffer to load into memory (our dataset is too large to fit entirely into memory). 

For example:  

```
python CNN_train.py -d F:\tf_dataset -o F:\training_output -e 100 -bu 8000 -ba 400
```


During training, the trained CNN and a log file will be stored at the **output_path**. 


Model conversion and its integration in VTM
-------------------------------------------


We utilize the [frugally-deep](https://github.com/Dobiasd/frugally-deep/tree/67a8fbce938353cde316d97f70c030172e50915e) library, which is a header-only library for using Keras (TensorFlow) models
in C++. With this library, we load the trained and converted CNN model into the VTM encoder, and the inference is performed on the CPU in real-time with C++. 


-Firstly, a conversion is required to 
convert the trained CNN from the h5 format into the json format. To acheive this, run the **convert_model.py** script located in **python_script\cnn_training**. For example:

```
python convert_model.py cnn_ori.h5 cnn_converted.json
```


-Secondly, we load the CNN model by providing its path to the VTM encoder via the command line. Specifically, we should call the encoder in the following manner:

```
.\EncoderApp -c <RAGOP32 config file> -cnn <location of the json file with file name>  -skipqt <0 or 1>   -thm <threshold>  -i <yuv input>  -wdt <frame width> -hgt <frame height>  -fr <frame rate> -f <num frames to encode> -q <QP value> -b <bin file> -o <rec yuv> -dph 1  -v 6
```


where the **-skipqt** corresponds to the **QTskip** in Figure 10 in the paper and the **-thm** is the threshold **Thm** of Algorithm 1.



