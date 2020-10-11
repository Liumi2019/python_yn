import tensorflow as tf
import tensorflow.keras.layers as k_layers
import numpy as np


class conv_deconv(k_layers.Layer):

    def __init__(self, filters, ksize=(3, 3), strides=(1, 1, 1, 1), padding='SAME', name='conv1'):
        super(conv_deconv, self).__init__(name=name)
        self.filters = filters
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        kernel_shape = tf.TensorShape((self.ksize[0], self.ksize[1], input_shape[-1], self.filters))

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=tf.initializers.he_normal(),
                                      trainable=True)

    def call(self, inputs, training=False):
        output = tf.nn.conv2d(inputs, filters=self.kernel, strides=self.strides, padding=self.padding)
        return output

    # @tf.function
    # def compute_output_shape(self, input_shape):
    #     shape = tf.TensorShape(input_shape).as_list()
    #     shape[-1] = self.filters
    #     return tf.TensorShape(shape)

    def deconv(self, inputs, shape):
        Img = tf.nn.conv2d_transpose(inputs, self.kernel, output_shape=shape,
                                     strides=self.strides, padding='SAME')

        return Img

    def de_maxpooling(self, img, max_ind, in_shape=(1, 32, 32, 8), out_shape=(1, 32, 32, 8)):

        img = tf.reshape(img, (-1, 1))
        max_ind = tf.reshape(max_ind, (-1, 1))
        # Img_L = tf.constant([out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]], dtype=tf.int64)
        print('#########################################')
        print(img[5, 0])
        print(max_ind[6, 0])

        # img = tf.constant(img)
        # max_ind = tf.constant(max_ind)
        Img_L = tf.constant([8192, 1], dtype=tf.int64)

        Img = tf.scatter_nd(max_ind, img, Img_L)
        # Img = tf.scatter_nd(indices, updates, Img_L)

        # in_W, in_H, in_C = in_shape[1], in_shape[2], in_shape[3]
        # Img_L = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
        # Img = tf.Variable(tf.zeros([1, Img_L]))
        #
        # for c in range(in_C):
        #     for x in range(in_H):
        #         for y in range(in_W):
        #             ind = max_ind[0, y, x, c]
        #             Img[0, ind] = img[0, y, x, c]

        return tf.reshape(Img, out_shape)
