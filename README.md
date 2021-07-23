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
  *      
       ```python
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
  ![image](https://user-images.githubusercontent.com/60976631/122643363-0f93a580-d12d-11eb-8168-e180455b14b8.png)
  
  *
       ```python
      import tensorflow as tf
      print(tf.test.is_built_with_cuda())
  
  ![image](https://user-images.githubusercontent.com/60976631/122643410-4ec1f680-d12d-11eb-8ba6-024de83acf1d.png)
  
  * 
      ```python
      import numpy as np
      from tensorflow import keras
      from tensorflow.keras import layers
      num_classes = 10
      input_shape = (28, 28, 1)
      # the data, split between train and test sets
      (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
      # Scale images to the [0, 1] range
      x_train = x_train.astype("float32") / 255
      x_test = x_test.astype("float32") / 255
      # Make sure images have shape (28, 28, 1)
      x_train = np.expand_dims(x_train, -1)
      x_test = np.expand_dims(x_test, -1)
      print("x_train shape:", x_train.shape)
      print(x_train.shape[0], "train samples")
      print(x_test.shape[0], "test samples")
      # convert class vectors to binary class matrices
      y_train = keras.utils.to_categorical(y_train, num_classes)
      y_test = keras.utils.to_categorical(y_test, num_classes)
      model = keras.Sequential(
          [
              keras.Input(shape=input_shape),
              layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2)),
              layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
              layers.MaxPooling2D(pool_size=(2, 2)),
              layers.Flatten(),
              layers.Dropout(0.5),
              layers.Dense(num_classes, activation="softmax"),
          ]
      )
      model.summary()
      batch_size = 128
      epochs = 5
      model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
      model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
  
 Good news if this code complete training in 12-13 sec
- IF you see all this congratulation you have enable CUDA in your **Nvidia Geforce GTX1650 TI**

