# rangle
Tensorflow model for estimating angles from images based on `tf.estimator.Estimator`s.

## Overview
Each model is identified by a `model_name`. Parameters are defined in `params/model_name.json`. Each model follows a typical convolution > dense layer chain, with optional dropout and batch normalization. The output for each example is either a probability distribution representing confidences of a given angle, or an angle in the range `(-pi, pi]`.

## Main files
- `estimator.py`: provides the `ModelBuilder` class for building various parts of the graph, along with `get_estimator` and `get_model_dir` for use with `tf.estimator.Estimator`
- `data.py`: provides data io functions.
- `server.py`: example of how to use for custom predictions. Note this doesn't use tensorflow's `TfServing`, so should be treated as an example only.
- `train.py`: training script.
- `losses.py`: provides losses used.

## How to use
1. Change the `load_data` function in `data.py` to something meaningful. Current implementation is random input for debugging purposes.
2. Define a model by creating a new `.json` file in `params` (or use `base`)
3. Train: `python train.py my_model_name -b 64 -s 10000000` trains with batch size of 64 and maximum number of steps 10000000.

## General model architecture
(conv (-> batch_norm)? -> `tf.nn.relu` (-> dropout)?)* -> flatten -> (dense (-> batch_norm)? -> `tf.nn.relu` (-> dropout)?)* (-> atan)?

## Params description
See `params/base.json` for an example parameterization of the models.
- `conv_kernels`: list of ints, e.g. [3, 3, 3]. Kernel size for each convolutional layer
- `conv_filters`: list of ints, e.g. [32, 64, 64]. Number of filters for each convolutional layer
- `padding`: one of ['VALID', 'SAME']. Used in `tf.layers.conv2d` and `tf.layers.max_pool2d`.
- `dense_nodes`: list of ints, e.g. [1024, 128]. Number dense nodes after convolutional layers and before the final fully connected layer.
- `target`: one of ['angle', 'distribution']. Indicates whether the output of the nextwork should be an angle (in which case the network infers an (x, y) value and calculates the angle via `tf.atan2`) or a distribution (see `n_bins`).
- `n_bins`: int, only applicable if `target == 'distribution'`. Number of bins to divide the range `(-pi, pi]` into.
- `dropout_rate`: float in [0, 1). Dropout rate used after each convolution/dense layer. Ignored if 0.
- `use_batch_norm`: bool indicating whether or not to use batch_norm. If true, it is applied before the activation (`relu`) and dropout.


## Potential changes
- Use `cosine` loss
- Move dropout to before `batch_norm`
- Different activation function (from `tf.nn.relu`)
- More convolutions between each `max_pool2d`
- Average pooling, or strided convolutional layers.
