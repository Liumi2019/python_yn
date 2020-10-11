import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
import numpy as np


# 卷积层332 -> 5*5
class Conv332(k_layers.Layer):
    def __init__(self, filters, name=None):
        super(Conv332, self).__init__(name=name)
        self.conv1 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bitchnm = k_layers.BatchNormalization()

    @tf.function
    def call(self, x, training):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.bitchnm(x, training=training)

        return x


# 卷积层333 -> 7*7
class Conv333(k_layers.Layer):
    def __init__(self, filters, name=None):
        super(Conv333, self).__init__(name=name)
        self.conv1 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bitchnm = k_layers.BatchNormalization()

    @tf.function
    def call(self, x, training):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.bitchnm(x, training=training)

        return x


# 卷积层334 -> 9*9
class Conv334(k_layers.Layer):
    def __init__(self, filters, name=None):
        super(Conv334, self).__init__(name=name)
        self.conv1 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv2 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv3 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same')
        self.conv4 = k_layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bitchnm = k_layers.BatchNormalization()

    @tf.function
    def call(self, x, training=True):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.bitchnm(x, training=training)

        return x


# 模型
class VGGNet16(k_models.Model):
    def __init__(self, input_shape, nclass):
        super(VGGNet16, self).__init__()
        self.inputlayer = k_layers.InputLayer(input_shape=input_shape, name='inputlayer')
        self.Conv1 = Conv332(8, name='conv332_1')
        self.Conv2 = Conv332(16, name='conv332_2')
        self.Conv3 = Conv333(32, name='conv333_1')
        self.Conv4 = Conv333(64, name='conv333_2')
        self.Conv5 = Conv333(64, name='conv333_3')

        self.maxPlooing = k_layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.flatten = k_layers.Flatten(name='flatten')
        self.dense = k_layers.Dense(nclass, activation='softmax', name='outputlayer')

    @tf.function
    def call(self, x, training):
        x = self.inputlayer(x)
        x = self.Conv1(x, training)
        x = self.Conv2(x, training)
        x = self.maxPlooing(x)
        x = self.Conv3(x, training)
        x = self.Conv4(x, training)
        x = self.Conv5(x, training)
        x = self.maxPlooing(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

    @tf.function
    def predict(self, x):
        if len(x.shape) < 4:
            x = x[np.newaxis, :, :, :]

        return self.call(x)

    @tf.function
    def showNet(self, x, layer_id):

        training = False
        x = self.Conv1(x, training)
        x1 = x
        x = self.Conv2(x, training)
        x = self.maxPlooing(x)
        x2 = x
        x = self.Conv3(x, training)
        x3 = x
        x = self.Conv4(x, training)
        x4 = x
        x = self.Conv5(x, training)
        x = self.maxPlooing(x)
        x5 = x

        if layer_id == 'Conv1':
            out_x = x1

        elif layer_id == 'Conv2':
            out_x = x2

        elif layer_id == 'Conv3':
            out_x = x3

        elif layer_id == 'Conv4':
            out_x = x4

        elif layer_id == 'Conv5':
            out_x = x5

        else:
            print('参数错误！')
            out_x = 0

        return out_x
