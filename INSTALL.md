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
--sudo apt-get install nvidia-390-- #2YP machines
sudo apt-get install nvidia-driver-430
```

Reboot machine: 
```
sudo reboot now
```

check that you have the correct driver:

`nvidia-smi`

if you get `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.` install b) above (430 driver?)


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


#### Update August 18, 2019 on Ubuntu 18.04 (bionic beaver) on Nvidia 430: to install CUDA 10 (first purge Blacklist for Nouveau Driver)

Follow [here](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07) to blacklist nouveau driver:
```
Create a file at /etc/modprobe.d/blacklist-nouveau.conf with the following contents:

blacklist nouveau
options nouveau modeset=0

```
And reboot. Then [deinstall old (assumes you skip step 2 b above](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07) and install:

```
sudo apt-get purge nvidia*

# Note this might remove your cuda installation as well
sudo apt-get autoremove 
```

```
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run
```

accept (and choose all) - install 418

Afterwards, set the paths:

```
Please make sure that
 -   PATH includes /usr/local/cuda-10.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.1/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.1/doc/pdf for detailed information on setting up CUDA.
Logfile is /var/log/cuda-installer.log
```

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

Note: tensorflow does NOT work on python3.7. Make sure to use python 3.6. For this reason we create a conda environment and load it afterwards.

`conda create -n py36tfnew python=3.6 anaconda`


and activate it

`conda activate py36tfnew`

Make sure python 3.6 is loaded, by typing

`python`

and getting

```
Python 3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, 19:16:44)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```


Modify and add the following two lines to your `.bashrc` file to set the `PATH` to point to the cuda-9.0/lib64 (see above) 
and also `LD_LIBRARY_PATH`, i.e.,
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
```

## 9. Install keras with tensorflow

Make sure to choose version 1.9, don't use conda install but use pip as [1] does, and do not use `keras-gpu` (not: `conda install -c anaconda keras-gpu`, it uses to new CUDA drivers, got a mismatch). Instead we follow Step 3. Install Tensorflow with Gpu support in [2] by N.Fridman but use 1.9:

```
conda activate py36tfnew
pip install tensorflow-gpu==1.9
```

#### Update Aug 2019: use tensorflow 1.14 for CUDA 10 (1.9 does work for CUDA 9)
See list [here](https://www.tensorflow.org/install/source).

```
pip install tensorflow-gpu==1.14
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

## 11. Cleaning up

Remove the anaconda installer. 

```
rm Anaconda3-2018.12-Linux-x86_64.sh
```

# References:

* [1] [Taylor Denouden. Installing Tensorflow GPU on Ubuntu 18.04 LTS](https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138)
* [2] [Naomi Fridman. Install Conda TensorFlow-gpu and Keras on Ubuntu 18.04](https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25)

