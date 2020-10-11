import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
import myLayer.conv_deconv as de_conv
import numpy as np


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
        self.conv1 = de_conv.conv_deconv(8, ksize=(3, 3), strides=(1, 1, 1, 1), padding='SAME')
        self.maxpool_arg = arg_maxpool(ksize=(2, 2), strides=(2, 2), padding='VALID')
        self.flten = k_layers.Flatten()
        self.fc = k_layers.Dense(nclass, activation='softmax')

    @tf.function
    def call(self, x, training=False):
        x = self.inputlayer(x)
        x = self.conv1(x)
        x, _ = self.maxpool_arg(x)
        x = self.flten(x)
        x = self.fc(x, training=training)

        return x

    @tf.function
    def showNet(self, x, layer_id):

        training = False
        x = self.inputlayer(x)
        shape1 = (1, 32, 32, 3)
        x = self.conv1(x, training=training)
        shape2 = (1, 32, 32, 8)
        x, max_ind = self.maxpool_arg(x)
        shape3 = (1, 16, 16, 8)
        x1 = x

        x1 = self.conv1.de_maxpooling(x1, max_ind, in_shape=shape3, out_shape=shape2)
        x2 = self.conv1.deconv(x1, shape1)

        if layer_id == 'conv1':
            out_x = x2
        else:
            out_x = 0

        return x1, out_x

    # def arg_maxpooling(self, img, max_ind, in_shape=(1, 32, 32, 8), out_shape=(1, 32, 32, 8)):
    #
    #     print(img[0, :, :, 0])
    #     img = np.array(img)
    #     in_W, in_H, in_C = in_shape[1], in_shape[2], in_shape[3]
    #     Img_L = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    #     Img = np.zeros((1, Img_L))
    #
    #     for c in range(in_C):
    #         for x in range(in_H):
    #             for y in range(in_W):
    #                 ind = max_ind[0, y, x, c]
    #                 Img[0, ind] = img[0, y, x, c]
    #
    #     return tf.reshape(Img, out_shape)

