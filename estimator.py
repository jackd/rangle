import os
import json
import tensorflow as tf
import numpy as np
from losses import angle_diff, earth_mover_loss


_root_dir = os.path.dirname(os.path.realpath(__file__))
_params_dir = os.path.join(_root_dir, 'params')
_models_dir = os.path.join(_root_dir, 'models')


class ModelBuilder(object):
    """Class for assisting in the building of `model_fn` for Estimator."""

    def __init__(self, params):
        """
        Construct the model_builder based on params.

        Params should be:
            conv_kernels: list of nc kernel sizes (int) for each conv layer
            conv_filters: list of nc filters/channels (int) for each conv layer
            padding: one of SAME/VALID used for conv/max pooling
            dense_nodes: list of nd nodes (int) for each fully connected layer
            target: one of 'angle' or 'distribution'
            dropout_rate: rate of dropout. No dropout applied if 0.
            use_batch_norm: bool indicating whether or not to use batch_norm
            n_bins: number of bins if target is 'distribution'. Ignored/
                not required if target is 'angle'.
        """
        self.params = params

    def get_inference(self, image, training):
        if self.params['use_batch_norm']:
            def activation(x):
                return tf.nn.relu(tf.layers.batch_normalization(
                    x, training=training, scale=False))
        else:
            activation = tf.nn.relu

        x = image
        padding = self.params['padding']
        dropout_rate = self.params['dropout_rate']
        initializer = tf.contrib.layers.xavier_initializer()
        # conv layers
        for k, n in zip(self.params['conv_kernels'],
                        self.params['conv_filters']):
            x = tf.layers.conv2d(
                x, n, k, padding=padding, activation=activation,
                kernel_initializer=initializer)
            if dropout_rate > 0:
                x = tf.layers.dropout(x, rate=dropout_rate, training=training)
            x = tf.layers.max_pooling2d(x, 3, 2, padding=padding)
        # fc layers
        x = tf.contrib.layers.flatten(x)
        for n in self.params['dense_nodes']:
            x = tf.layers.dense(x, n, activation=activation)
            if dropout_rate > 0:
                x = tf.layers.dropout(x, rate=dropout_rate, training=training)

        target = self.params['target']
        if target == 'angle':
            x = tf.layers.dense(x, 2)
            x, y = tf.unstack(x, axis=-1)
            inference = tf.atan2(y, x)
            return inference
        elif target == 'distribution':
            n_bins = self.params['n_bins']
            dist = tf.layers.dense(x, n_bins, activation=tf.nn.softmax)
            return dist
        else:
            raise Exception(
                'target key must be one of "target" or "distribution"')

    def get_loss(self, inference, labels):
        target = self.params['target']
        if target == 'angle':
            loss = tf.nn.l2_loss(angle_diff(inference - labels))
            tf.summary.scalar('angle_loss', loss)
        elif target == 'distribution':
            n_bins = self.params['n_bins']
            self._angles = tf.constant(
                np.linspace(-np.pi, np.pi, n_bins+1)[1:])
            loss = earth_mover_loss(inference, self._angles, labels)
        else:
            raise Exception(
                'target key must be one of "target" or "distribution"')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) > 0:
            reg_loss = tf.add_n(reg_losses)
            tf.summary.scalar('reg_loss', reg_loss)
            loss += reg_loss
            tf.summary.scalar('total_loss', loss)
        return loss

    def get_prediction(self, inference):
        target = self._params['target']
        if target == 'angle':
            return inference
        elif target == 'distribution':
            return self._angles.gather(tf.argmax(inference, axis=-1))

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer()
        steps = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
        if len(steps) == 1:
            step = steps[0]
        else:
            raise Exception('Multiple global steps disallowed')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # The following forces update_ops created elsewhere
        # (e.g. batch_norm stuff) to be run whenever train_op is run.
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, step)
        return train_op

    def model_fn(self, features, labels, mode, config):
        image = features
        if mode == tf.estimator.ModeKeys.TRAIN:
            inference = self.get_inference(image, training=True)
            loss = self.get_loss(inference, labels)
            train_op = self.get_train_op(loss)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)
        else:
            inference = self.get_inference(image, training=False)
            prediction = self.get_prediction(inference)
            if mode == tf.estimator.ModeKeys.EVAL:
                loss = self.get_loss(inference, labels)
                error = tf.abs(angle_diff(prediction - labels))
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops={'error': error})
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode, predictions=prediction)


def _load_params(model_name):
    path = os.path.join(_params_dir, '%s.json' % model_name)
    with open(path, 'r') as f:
        params = json.load(f)
    return params


def _model_fn(features, labels, mode, params, config):
    return ModelBuilder(params).model_fn(features, labels, mode, config)


def get_model_dir(model_name):
    return os.path.join(_models_dir, model_name)


def get_estimator(model_name='base', config=None):
    params = _load_params(model_name)
    model_dir = get_model_dir(model_name)
    return tf.estimator.Estimator(
        _model_fn, model_dir, params=params, config=config)
