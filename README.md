# mnist-in-mpi
Handwritten digits recognition using a parallel neural network.

<p align="center">
  <img src="https://github.com/stefan1niculae/mnist-in-mpi/raw/master/frontend/demo.gif" alt="Demo Gif"/>
  <img src="https://github.com/stefan1niculae/mnist-in-mpi/raw/master/doc/time-chart.png" alt="Time Chart"/>
  <img src="https://github.com/stefan1niculae/mnist-in-mpi/raw/master/doc/train-chart.png" alt="Training Chart"/>
  <img src="https://github.com/stefan1niculae/mnist-in-mpi/raw/master/doc/paper-sample.png" alt="Paper Sample"/>
</p>


## Build

* Generate make file: `cmake CMakeLists.txt`
* Compile: `make`

## Dataset
```
mkdir data
wget -O data/train.csv http://pjreddie.com/media/files/mnist_train.csv
wget -O data/test.csv http://pjreddie.com/media/files/mnist_test.csv
```
Or download manually from http://pjreddie.com/projects/mnist-in-csv and place them in `data/train.csv` and `data/test.csv`.

## Run

* Make sure `models` and `histories` folders exist: `mkdir models histories`
* Train a model: `mpirun -np <number of processes> ./mnist_in_mpi train`
* Start the server: `./server.py`
* Visit [localhost:5000](http://localhost:5000) and try it out!
