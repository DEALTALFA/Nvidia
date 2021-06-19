# Nvidia

To enable CUDA 

* check tensorflow corresponding from [here](https://www.tensorflow.org/install/source_windows)
 ![image](https://user-images.githubusercontent.com/60976631/122642620-f7218c00-d128-11eb-876b-1768dc3f8fa0.png)

* After some hit and trail .I got to know for me tensorflow==2.1,tensorflow-gpu==2.1,cuDNN 7.6.5 with CUDA 10.1 worker for me.

## Configuration
- Download CUDA 10.1(CUDA Toolkit 10.1 update2) from [here](https://developer.nvidia.com/cuda-toolkit-archive)
or search cuda 10.1 download->legacy Release->find your version
- Download cuDNN compatiable with you CUDA from [here](https://developer.nvidia.com/rdp/cudnn-archive)
  * **Recommended** Downloaded cuDNN v7.6.5 (November 5th, 2019) for CUDA 10.1 in my case 
  * Also worked with cuDNN v7.6.0 (May 20, 2019), for CUDA 10.1
- Download Microsoft Visual Community 2019 and check both of this  ![image](https://user-images.githubusercontent.com/60976631/122642464-22f04200-d128-11eb-8621-13f66b564998.png)
   Downloading both will work haven't confirm have after installing both it worked. Have to test without mobile does it work???
- Download anaconda 
## Installation
- Install CUDA 10.1
- extract the cuDNN v7.6.5 zip file and copy the files that are inside the each folder to the path were you installed the CUDA to that repective folder
  For more detail check out [ this ](https://medium.com/analytics-vidhya/installing-tensorflow-cuda-cudnn-for-nvidia-geforce-gtx-1650-ti-onwindow-10-99ca25020d6c) and other one [this](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-install/index.html#install-windows) ![image](https://user-images.githubusercontent.com/60976631/122642662-351eb000-d129-11eb-9936-f42d54010e97.png)
- I also installed Microsoft Visual C++ Redistributable for Visual Studio 2019 don't know will it effect or not from [here](https://visualstudio.microsoft.com/downloads/) under Other tools,Framwork.....
- Install anaconda 
  * create new environment with `conda create -n dl python=3.7` NOTE: python version is according to [this doc](https://www.tensorflow.org/install/source_windows)
   ![image](https://user-images.githubusercontent.com/60976631/122643093-731cd380-d12b-11eb-98b6-b5e645658c0e.png)
  * then run this  commands `conda install tensorflow==2.1` and `conda install tensorflow-gpu` 
-To check wherether the we have enable GPU for deep learning by enabling CUDA run any of the below code
       
       ```python
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())```
  ![image](https://user-images.githubusercontent.com/60976631/122643363-0f93a580-d12d-11eb-8168-e180455b14b8.png)

      ```python
      import tensorflow as tf
      print(tf.test.is_built_with_cuda())```
  ![image](https://user-images.githubusercontent.com/60976631/122643410-4ec1f680-d12d-11eb-8ba6-024de83acf1d.png)
 - IF you see all this congratulation you have enable CUDA in your **Nvidia Geforce GTX1650 TI**

