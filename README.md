CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields
============================================================

This is the source code of paper **CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields** currently under review of IEEE Transaction of Image Processing.

Our dataset MVF-Inter is available at https://1drv.ms/f/s!Aoi4nbmFu71Hgx9FJphdskXfgIVo?e=fXrs0o


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

For VS 2017 use "Visual Studio 15 2017 Win64", for VS 2019 use "Visual Studio 16 2019".

Visual Studio 2019 also allows you to open the CMake directory directly. Choose "File->Open->CMake" for this option.

For the release build of this project in Visual Studio, we should set the EncoderApp as Startup Project. Beforing building it, 
right click on the EncoderLib and set the "Treat Warnings As Error" to No in "Property->C/C++/General". Then add the compile option 
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


Training instructions
---------------------

For the training of the related CNN network in the paper, plz download the aforementioned TFrecord dataset. Before start training, package of Keras, numpy and Tensorflow GPU 
are required in the environment of training. Then we can call the scripts in folder cnn_script to train the CNN. The python script CNN_train.py is for defining the CNN structure
and launching its training. 
 
This script should be called in the following manner: python CNN_train.py -d <dataset_path> -o <output_path> -e <epoch> -bu <buffer_size> -ba <batch_size>
where the buffer_size indicate the size of buffer to load in the memory (our dataset is too large to fit in the memory). 

An example:  python CNN_train.py -d F:\tf_dataset -o F:\training_output -e 100 -bu 8000 -ba 400

During the training, the trained CNN and a logfile are stored at the output_path. 




