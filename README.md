# mnist-in-mpi
Handwritten digits recognition using a parallel neural network.

TODO: redo the demo gif with a white background, move it to frontend\
<p align="center">
  <img src="https://github.com/stefan1niculae/mnist-in-mpi/raw/master/demo.gif" alt="Demo gif"/>
</p>

TODO: add training accuracy plot & parallel performance plot

TODO: add paper pdf and a small screenshot


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
* Train a model: `./mnist_in_mpi train`
* Start the server: `./server.py`
* Visit [localhost:5000](http://localhost:5000) and try it out!
