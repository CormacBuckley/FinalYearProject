Bootstrap:docker  
From:ubuntu:latest  

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
  pip install tensorflow-gpu
  pip install numpy
  pip install scipy
  pip install Pillow
  pip install opencv-python
  pip install h5py
  pip install matplotlib
  pip install scikit-image
  pip install cython
  pip install keras

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success