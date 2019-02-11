Bootstrap: docker
From: tensorflow/tensorflow:1.8.0-gpu-py3

%environment
  # use bash as default shell
  SHELL=/bin/bash
  export SHELL

%setup
  # runs on host - the path to the image is $SINGULARITY_ROOTFS

%post
  # post-setup script

  # load environment variables
  . /environment

  # use bash as default shell
  echo 'SHELL=/bin/bash' >> /environment

  # make environment file executable
  chmod +x /environment

  # default mount paths
  mkdir /scratch /data 

  # additional packages
  apt-get update
  apt-get install -y python-tk
  apt-get install -y libsm6 libxext6 
  apt-get install python3-numpy python3-scipy
  apt-get install python3-pil
  apt-get install python-opencv
  apt-get install libhdf5-dev
  apt-get build-dep python-matplotlib
  apt-get install build-essential cython3
  apt-get install python-pip python-dev build-essential 
  pip install keras

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success
