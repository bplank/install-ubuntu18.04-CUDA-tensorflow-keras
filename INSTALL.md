### Final working configuration:

- Ubuntu 18.04
- nvidia-390 drivers forom PPA repo
- CUDA 9.0 (downloaded for Ubuntu 17.04 version; install but make sure to ignore driver)
- make sure conda bin is added to PATH and LD_LIBRARY_PATH is updated as well (before proceeding to install Python packages)
- Anaconda Python 3.6 (make sure it is not 3.7)
- tensorflow1.9 (make sure it is not tensorflow1.12); install tensorflow via pip (not conda) and do not use keras-gpu to get tensorflow
- pip install keras

# Step-by-step instructions

## 1. Install Ubuntu 18.04

`lsb_release -a`

shows which Ubuntu release is installed

## 2. Install NVIDIA drivers

#### a) First, detect the model of your nvidia graphic card and the recommended driver. To do so execute:

`ubuntu-drivers devices`

If ubuntu-drivers is not found, run the following command in terminal to install it:

`sudo apt-get install ubuntu-drivers-common`

#### b) install NVIDIA driver

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-390
```

Reboot machine

check that you have the correct driver:

`nvidia-smi`


## 3. Download CUDA from NVidia

https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux
Get CUDA from [NVIDIA](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704)

Get CUDA 9.0, select Linux, x86_64, Ubuntu, select the 17.04 version and "runfile(local)". See [1].

```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run

sudo sh cuda_9.0.176_384.81_linux.run --override
```

accept

yes (to unsupported configuration)

no (to install driver)

yes (to install toolkit)

yes (default location)


N.B. It has the 384 flag in the name, make sure to not install the new driver as we want to use the 390.

“Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?”. Make sure you *don’t* agree to install the new driver.  Choose *no*.

## 4. Check installation and add cuda to your path

Check if you see the cuda installation folder:

`ls /usr/local/cuda-9.0/`


Add Cuda to your PATH (otherwise we don't have nvcc)

`export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`

and then either set LD_LIBRARY_PATH, or 
add `/usr/local/cuda-9.0/lib64`
to
`/etc/ld.so.conf.d/*.conf`

check that we have:

`nvcc --version`

to get the CUDA version


## 5. Install cuDNN

These are the instructions from [1]:

"Next, head to https://developer.nvidia.com/cudnn to get CUDNN 7.0. Go
to the downloads archive page again and find version 7.0 for CUDA 9.0
that you just installed. Download the link that says “cuDNN v7.0.5
Library for Linux”. This will download an archive that you can unpack
and move the contents the correct locations."


```
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7 
tar -zxvf cudnn-9.0-linux-x64-v7.tgz`

# Move the unpacked contents to your CUDA directory
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp  cuda/include/cudnn.h /usr/local/cuda-9.0/include/
# Give read access to all users
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## 6. Do the CUDA post-install actions

So that Tensorflow can find your CUDA installation and use it properly, you need to add these lines to the end of you ~/.bashrc or ~/.zshrc.

`export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}`    (we already did this earlier)
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

(we skipped installing this diagnostic tool: Install libcupti in [1])

## 7. Install Anaconda

`wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh`

`sh Anaconda3-2018.12-Linux-x86_64.sh`

* Agree to the licence 
* Say 'yes' to adding the Anaconda config to `.bashrc`
* say 'no' to some Microsoft VSCode

Make sure from now on you are loading your own Python Anaconda version, i.e., load your `.bashrc` file with:

```
source .bashrc
```

(in case you skip this and the wrong python is loaded, you might get the following error in the next step):
```
conda create -n py36tfnew python=3.6 anaconda

CondaValueError: prefix already exists: /home/sebastian/anaconda3/envs/py36tfnew
```

## 8. Create a python 3.6 environment

Note: tensorflow does NOT work on python3.7

`conda create -n py36tfnew python=3.6 anaconda`


and activate it

`conda activate py36tfnew`

Make sure python 3.7 is loaded, by typing

`python`

and getting

```
python
Python 3.7.1 (default, Dec 14 2018, 19:28:38)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```


set the `PATH` to point to the cuda-9.0/lib64 (see above) - Ideally add it to `.bashrc`

and also
`
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
`

## 9. Install keras with tensorflow

Make sure to choose version 1.9, don't use conda install but use pip as [1] does, and do not use `keras-gpu` (not: `conda install -c anaconda keras-gpu`, it uses to new CUDA drivers, got a mismatch). Instead we follow Step 3. Install Tensorflow with Gpu support in [2] by N.Fridman but use 1.9:

```
conda activate py36tf
pip install tensorflow-gpu==1.9
```


## 10. Install Keras and Test installation


Now we install keras
```
pip install keras
```

(instead we do not use `conda install -c conda-forge keras`)

Now we can test the installation, which should show the GPU:
```
python
>> from keras import backend as K 
>> K.tensorflow_backend._get_available_gpus()
```



# References:

* [1] [Taylor Denouden. Installing Tensorflow GPU on Ubuntu 18.04 LTS](https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138)
* [2] [Naomi Fridman. Install Conda TensorFlow-gpu and Keras on Ubuntu 18.04](https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25)

