import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
import numpy as np


# 创建一个3*3的卷积层
class Conv3_3(k_layers.Layer):
    def __init__(self, filters, name=None):
        super(Conv3_3, self).__init__(name=name)
        self.conv1 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpooling = k_layers.MaxPool2D((2, 2), strides=(2, 2))
        self.bithnmlize = k_layers.BatchNormalization()

    @tf.function
    def call(self, x, training):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.maxpooling(x)
        x = self.bithnmlize(x, training=training)

        return x


# 测试模型
class VGGNet(k_models.Model):
    def __init__(self, input_shape, nclass):
        super(VGGNet, self).__init__()
        self.inputlayer = k_layers.InputLayer(input_shape=input_shape, name='inputlayer')
        self.Conv1 = Conv3_3(8, name='conv1')
        self.Conv2 = Conv3_3(16, name='conv2')
        self.flatten = k_layers.Flatten(name='flatten')
        self.dense = k_layers.Dense(nclass, activation='softmax', name='outputlayer')

    @tf.function
    def call(self, x, training):
        x = self.inputlayer(x)
        x = self.Conv1(x, training)
        x = self.Conv2(x, training)

        x = self.flatten(x)
        x = self.dense(x)

        return x

