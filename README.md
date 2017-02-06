# mnist-in-mpi
Parallel handwritten digits recognition using a neural network.

## Installation
* Generate make file: `cmake CMakeLists.txt`
* Build: `make`

## Dataset
```
mkdir data
wget -O data/train.csv http://pjreddie.com/media/files/mnist_train.csv
wget -O data/test.csv http://pjreddie.com/media/files/mnist_test.csv
```
Or download manually from http://pjreddie.com/projects/mnist-in-csv and place them in `data/train.csv` and `data/test.csv`.

## Run
* Make sure `models` and `histories` folders exist: `mkdir models histories`
* Start: `./mnist_in_mpi.out`
