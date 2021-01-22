import tensorflow as tf
import numpy as np
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc


def cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    # layer_1 = activ(conv(scaled_images, 'c1', n_filters=32,
    #                      filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=64,
                         filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64,
                         filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64,
                         filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    fc1 = activ(linear(layer_3, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
    fc2 = activ(linear(fc1, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return fc2
