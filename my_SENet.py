from scipy.io import loadmat
import numpy as np

from cifar10 import *

import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, GlobalAvgPool2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Input, Activation, ReLU

from tensorflow.keras import Model

# 全局变量
model_para = {"bitch_size": 32, "class_num": 2}
blocks = 3  # 卷积层的个数
reduction_ratio = 4
cardinality = 8
depth = 64
test_iteration = 50


class MyFirstLayer(Model):
    def __init__(self, filters, kernel_size, stride):
        super(MyFirstLayer, self).__init__(name="firstlayer")
        self.conv1 = Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=stride, padding='same', input_shape=(32, 32, 3))
        self.bitchnorm = BatchNormalization()
        self.relu = ReLU()

    @tf.function
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.bitchnorm(x, training=training)
        x = self.relu(x)

        return x


class MySplitLayer(Model):
    def __init__(self, outdim, depth=64, stride=(1, 1), cardinality=8, block=3, ratio=4):
        super(MySplitLayer, self).__init__(name="splitlayer")
        self.cardinality = cardinality
        self.block = block
        self.outdim = outdim
        self.ratio = ratio
        self.depth = depth

        self.conv1 = Conv2D(filters=self.depth, kernel_size=(1, 1), strides=stride, padding='same')
        self.conv2 = Conv2D(filters=self.depth, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2_2 = Conv2D(filters=self.depth, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.convout = Conv2D(filters=outdim, kernel_size=(1, 1), strides=stride, padding='same')

        self.dens1 = Dense(units=self.outdim / self.ratio)
        self.dens2 = Dense(units=self.outdim)

        self.bitchnorm = BatchNormalization()
        self.argpool = AveragePooling2D()
        self.relu = ReLU()

    @tf.function
    def call(self, inputs, training=False):

        for i in range(self.block):

            input_dim = int(np.shape(inputs)[-1])
            if input_dim * 2 == self.outdim:
                flag = True
                channel = input_dim // 2
            else:
                flag = False
                channel = input_dim // 2

            layers_split = list()

            for ii in range(self.cardinality):
                x = self.conv1(inputs, training=training)
                x = self.bitchnorm(x, training=training)
                x = self.relu(x)

                if flag:
                    x = self.conv2_2(x, training=training)
                else:
                    x = self.conv2(x, training=training)

                x = self.bitchnorm(x, training=training)
                x = self.relu(x)
                layers_split.append(x)

            x = tf.concat(layers_split, axis=3)
            print("》》》》》》》》》》》》》》X的维度>>>>>>>>>")
            print(x.shape)
            # 过渡层
            x = self.convout(x, training=training)
            print("》》X的维度")
            print(x.shape)
            x = self.bitchnorm(x, training=training)

            # se层
            se_x = GlobalAvgPool2D()(x)
            se_x = self.dens1(se_x)
            se_x = self.relu(se_x)
            se_x = self.dens2(se_x)
            excitation = tf.nn.sigmoid(se_x)

            excitation = tf.reshape(excitation, [-1, 1, 1, self.outdim])
            x = x * excitation

            if flag is True:
                pad_input_x = self.argpool(inputs)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = inputs

            inputs = self.relu(x + pad_input_x)

        return inputs


class MySENet(Model):
    def __init__(self, nclass):
        super(MySENet, self).__init__()
        self.nclass = nclass
        self.firstlayer = MyFirstLayer(filters=64, kernel_size=(3, 3), stride=(1, 1))
        self.split1 = MySplitLayer(outdim=64, depth=64)
        self.split2 = MySplitLayer(outdim=128, depth=64)
        self.split3 = MySplitLayer(outdim=256, depth=64)
        self.lastlayer = Dense(units=self.nclass)

    @tf.function
    def call(self, inputs, training=False):

        x = self.firstlayer(inputs, training=training)
        x = self.split1(x, training=training)
        x = self.split2(x, training=training)
        x = self.split3(x, training=training)

        x = GlobalAvgPool2D()(x)
        x = Flatten()(x)
        x = self.lastlayer(x)

        return x


if __name__ == "__main__":

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(1000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model = MySENet(nclass=10)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
