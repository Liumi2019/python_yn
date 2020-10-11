from STNet import transformer

import os

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from cifar10 import *

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 10


class STNet(Model):
    def __init__(self, nclass, stn=False):
        super(STNet, self).__init__(name="stnet")
        self.nclass = nclass
        self.stn = stn

        self.conv1 = layers.Conv2D(filters=8, kernel_size=(3, 3), name='conv1',
                                   strides=1, padding="valid", activation='relu')
        self.maxpool1 = layers.MaxPool2D(pool_size=(2, 2), name='conv1_maxpool')
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(3, 3), name='conv2',
                                   strides=1, padding="valid", activation='relu')
        self.maxpool2 = layers.MaxPool2D(pool_size=(2, 2), name='conv2_maxpool')
        self.flatten = layers.Flatten(name='flatten')
        self.dens1 = layers.Dense(units=64, activation='relu', name='dense1')
        self.dens2 = layers.Dense(units=self.nclass, activation='softmax', name='output')

        # stn层
        self.H, self.W, self.C = 32, 32, 3
        self.B = 32
        initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
        initial = initial.astype('float32')
        self.b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')

    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)

        if self.stn:
            x = transformer.spatial_transformer_network(x, self.b_fc1)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dens1(x)
        x = self.dens2(x)

        return x

    @tf.function
    def predict(self, x):
        if len(x.shape) < 4:
            x = x[np.newaxis, :, :, :]

        return self.call(x)

    @tf.function
    def showNet(self, x, layer_id):
        x = self.conv1(x)
        x = self.maxpool(x)

        if layer_id == 'conv1':
            out_x = x
        else:
            if self.stn:
                x = transformer.spatial_transformer_network(x, self.b_fc1)
                stn_x = x
            else:
                stn_x = 0

            if layer_id == 'stn':
                out_x = stn_x
            else:
                x = self.conv2(x)
                x = self.maxpool(x)
                if layer_id == 'conv2':
                    out_x = x
                else:
                    out_x = 0
                    print("没有这一层！")

        return out_x


def creat_ds(parament_all):

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    # img = test_x[10, :, :, :]
    # np.save('img.npy', img)
    bitchsize = parament_all['BitchSize']
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(10000).batch(bitchsize)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(bitchsize)

    return train_ds, test_ds


def model_run(train_ds, test_ds, model_path, parament_all):

    nclass = parament_all['nClass']
    stn = parament_all['stn']
    is_model_path = model_path + '.index'
    if os.path.exists(is_model_path):
        model = STNet(nclass, stn)
        model.load_weights(model_path)
        print("加载的模型")
    else:
        model = STNet(nclass, stn)
        print("新建一个模型")

    # 损失和优化
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # 打印的指标
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):  # 训练
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):  # 测试
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    result_data = np.zeros((4, 0))

    Epochs = parament_all['Epochs']
    for epoch in range(Epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # tf.summary.trace_on(graph=True, profiler=True)

        for images, labels in train_ds:
            train_step(images, labels)

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', train_loss.result(), step=epoch)
        #     tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        # with train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="my_func_trace",
        #         step=0,
        #         profiler_outdir=train_log_dir)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss', test_loss.result(), step=epoch)
        #     tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        temp_array = np.vstack((np.array(train_loss.result()), np.array(train_accuracy.result()),
                                np.array(test_loss.result()), np.array(test_accuracy.result())))

        result_data = np.hstack((result_data, temp_array))

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

    model.save_weights(model_path)

    np.save('LenetSEresult.npy', result_data)
    if parament_all['accShow']:
        plt.plot(np.arange(Epochs), result_data[0, :], 'b--')
        plt.plot(np.arange(Epochs), result_data[1, :], 'b')
        plt.plot(np.arange(Epochs), result_data[2, :], 'r--')
        plt.plot(np.arange(Epochs), result_data[3, :], 'r')
        plt.legend(['训练损失', '训练准确率', '测试损失', '测试准确率'])
        plt.show()


if __name__ == "__main__":
    save_model_path = '保存LeNet模型/SNT/weights.ckpt'

    parament_all = {'Epochs': 5, 'BitchSize': 32, 'accShow': False,
                    'nClass': 10, 'stn': True}

    train_ds, test_ds = creat_ds(parament_all)
    model_run(train_ds, test_ds, save_model_path, parament_all)

    stn = parament_all['stn']

    model = STNet(nclass=10, stn=stn)
    model.load_weights(save_model_path)

    img = np.load('img.npy')
    plt.imshow(img)
    plt.show()
    img = img[np.newaxis, :, :, :]
    a = model.predict(img)
    pre = tf.argmax(a, 1) + 1

    img_layers = model.showNet(x=img, layer_id='stn')

    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(img_layers[0, :, :, i])
    plt.show()
