import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models


class EEG1DConv(k_models.Model):

    def __init__(self, input_shape=(8, 400, 1), out_L=(1, 50)):
        super(EEG1DConv, self).__init__()

        self.inputlayer = k_layers.InputLayer(input_shape=input_shape, name='inputlayer')

        # 深度块
        self.conv1 = k_layers.Conv2D(8, (1, 50), strides=(1, 1), name='conv1', padding='same', activation='relu')
        self.conv2 = k_layers.Conv2D(16, (1, 20), strides=(1, 1), name='conv2', padding='same', activation='relu')
        self.conv3 = k_layers.Conv2D(32, (1, 10), strides=(1, 1), name='conv3', padding='same', activation='relu')

        self.maxpool1 = k_layers.MaxPool2D(pool_size=(1, 2), name='maxpool1', strides=(1, 2))
        self.maxpool2 = k_layers.MaxPool2D(pool_size=(1, 2), name='maxpool2', strides=(1, 2))
        self.maxpool3 = k_layers.MaxPool2D(pool_size=(1, 2), name='maxpool3', strides=(1, 2))

        # 浅层块
        self.conv21 = k_layers.Conv2D(16, (1, 50), strides=(1, 1), name='conv21', padding='same', activation='relu')
        self.maxpool21 = k_layers.MaxPool2D(pool_size=(1, 4), name='maxpool21', strides=(1, 4))

        self.conv4 = k_layers.Conv2D(1, (8, 1), strides=(1, 1), name='conv4', padding='valid', activation='relu')

        self.outlayer = k_layers.Reshape(out_L)

    @tf.function
    def call(self, x, training=True):

        x = self.inputlayer(x)

        # 深度块
        x1 = self.conv1(x, training=training)
        x1 = self.maxpool1(x1)
        x1 = self.conv2(x1, training=training)
        x1 = self.maxpool2(x1)
        x1 = self.conv3(x1, training=training)
        x1 = self.maxpool3(x1)
        # 插值
        x1_shape = x1.shape.as_list()
        x1 = tf.image.resize(x1, [8, x1_shape[2]*2], method='bicubic')

        # 浅层块
        x = self.conv21(x, training=training)
        x = self.maxpool21(x)

        x = tf.concat([x, x1], axis=3)
        x = self.conv4(x, training=training)
        x = self.outlayer(x)

        return x


