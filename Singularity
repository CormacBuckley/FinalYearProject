Bootstrap:docker  
From:ubuntu:latest  

%labels
MAINTAINER Vanessasaur
SPECIES Dinosaur

%environment
RAWR_BASE=/code
export RAWR_BASE

%runscript
echo "This gets run when you run the image!" 


%post  
echo "This section happens once after bootstrap to build the image."  
sudo easy_install pip
pip install numpy
pip install scipy
pip install Pillow
pip install opencv-python
pip install h5py
pip install matplotlib
pip install scikit-image
pip install cython
pip install keras