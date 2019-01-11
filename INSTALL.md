### Working configuration:

- Ubuntu 18.04
- nvidia-390 drivers forom PPA repo
- CUDA 9.0 (downloaded for Ubuntu 17.04 version; install but make sure to ignore driver)
- make sure conda bin is added to PATH and LD_LIBRARY_PATH is updated as well (before proceeding to install Python packages)
- Anaconda Python 3.6 (make sure it is not 3.7)
- tensorflow1.9 (make sure it is not tensorflow1.12); install tensorflow via pip (not conda) and do not use keras-gpu to get tensorflow
- pip install keras


1. Install Ubuntu 18.04

// to see which ubuntu release is installed:
lsb_release -a

2. Install NVIDIA drivers

a) First, detect the model of your nvidia graphic card and the recommended driver. To do so execute:

//ubuntu-drivers devices --> no such command

If ubuntu-drivers is not found, run the following command in terminal to install it:

sudo apt-get install ubuntu-drivers-common

b) install NVIDIA driver

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-390

-- reboot machine

check that you have the correct driver:

nvidia-smi


3. Download CUDA from NVidia

https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704

(get CUDA 9.0, select Linux, x86_64, Ubuntu, select the 17.04 version and "runfile(local)"


sudo sh cuda_9.0.176_384.81_linux.run --override

accept

yes (to unsupported configuration)

no (to install driver)

yes (to install toolkit)

yes (default location)


==> has the 384 flag in the name, make sure to not install the new driver:

“Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?”. Make sure you don’t agree to install the new driver. NO!

4. Check installation and add cuda to your path

check if you see the cuda installation folder:

ls /usr/local/cuda-9.0/


Add Cuda to your PATH (otherwise we don't have nvcc)

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

add /usr/local/cuda-9.0/lib64
to
/etc/ld.so.conf.d/*.conf

check that we have:

nvcc --version
to get your cuda version


5. Install cuDNN

Next, head to https://developer.nvidia.com/cudnn to get CUDNN 7.0. Go
to the downloads archive page again and find version 7.0 for CUDA 9.0
that you just installed. Download the link that says “cuDNN v7.0.5
Library for Linux”. This will download an archive that you can unpack
and move the contents the correct locations.

tar -zxvf cudnn-9.0-linux-x64-v7.tgz

# Move the unpacked contents to your CUDA directory
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo cp  cuda/include/cudnn.h /usr/local/cuda-9.0/include/
# Give read access to all users
sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

6. Do the CUDA post-install actions

So Tensorflow can find your CUDA installation and use it properly, you need to add these lines to the end of you ~/.bashrc or ~/.zshrc.

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}    (we already did this earlier)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

(we skipped installing this diagnostic tool: Install libcupti)

7. Install Anaconda

wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh


8. Create a python 3.6 env ==> does NOT work on python3.7

conda create -n py36tfnew python=3.6 anaconda


and activate it

conda activate py36tfnew

set the PATH to point to the cuda-9.0/lib64


9. Install keras with tensorflow

///conda install -c anaconda keras-gpu
-- did not work, it uses to new CUDA drivers, got a mismatch

follow Step 3. Install Tensorflow with Gpu support in [2] by N.Fridman
//conda install -c anaconda tensorflow-gpu
-- installed tensorflow1.12 which did not work

downgrade to tensorflow1.9

//conda install -c anaconda tensorflow-gpu==1.9
don't use conda install but use pip as [1] does

conda activate py36tf
pip install tensorflow-gpu==1.9


10. Install Keras and Test installation

//.$conda install -c conda-forge keras

pip install keras

python
>> from keras import backend as K 
>> K.tensorflow_backend._get_available_gpus()

set the path!!

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

Source:

[1] https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138
[2] https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25

