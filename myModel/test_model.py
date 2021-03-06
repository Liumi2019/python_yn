import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models


class arg_maxpool(k_layers.Layer):

    def __init__(self, ksize=(2, 2), strides=(2, 2), padding='VALID'):
        super(arg_maxpool, self).__init__()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    @tf.function
    def call(self, x):
        x, arg = tf.nn.max_pool_with_argmax(x, self.ksize, self.strides, self.padding)

        return x, arg


class test_model(k_models.Model):

    def __init__(self, input_shape, nclass):
        super(test_model, self).__init__()
        self.inputlayer = k_layers.InputLayer(input_shape=input_shape, name='inputlayer')
        self.conv1 = k_layers.Conv2D(8, (3, 3), (1, 1), padding='same', activation='relu')
        self.maxpool_arg = arg_maxpool(ksize=(2, 2), strides=(2, 2), padding='SAME')
        self.flten = k_layers.Flatten()
        self.fc = k_layers.Dense(nclass, activation='softmax')

    @tf.function
    def call(self, x, training):
        x = self.inputlayer(x)
        x = self.conv1(x, training=training)
        x, _ = self.maxpool_arg(x)
        x = self.flten(x)
        x = self.fc(x, training=training)

        return x

    @tf.function
    def showNet(self, x, layer_id):
        training = False
        x = self.inputlayer(x)
        x = self.conv1(x, training=training)
        x1 = x
        x, max_ind = self.maxpool_arg(x)
        x2 = x

        if layer_id == 'conv1':
            pre_maxpool = x1
            out_x = x2
        else:
            pre_maxpool = 0
            out_x = 0

        return pre_maxpool, out_x, max_ind