from scipy.io import loadmat
import numpy as np

from cifar10 import *

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, GlobalAvgPool2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Input, Activation

from tensorflow.keras import Model


# 全局变量
model_para = {"bitch_size": 32, "class_num": 2}
blocks = 3  # 卷积层的个数
reduction_ratio = 4
cardinality = 8
depth = 64
test_iteration = 50


def creat_dataSet(filename, model_para):

    (x_train, y_train), (x_test, y_test) = loadmat(filename)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(model_para["bitch_size"])
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(model_para["bitch_size"])

    return (x_train, y_train), (x_test, y_test)


# 定义卷积
def conv_layer(input, filter, kernel, stride, padding='same', layer_name="conv"):

    # 名字空间，相同的名字下，函数才可以调用
    with tf.name_scope(layer_name):
        network = tf.keras.layers.Conv2D(use_bias=False, filters=filter, kernel_size=kernel,
                                         strides=stride, padding=padding)(input)
        return network


def Global_Average_Pooling(x):
    return GlobalAvgPool2D(name='Global_avg_pooling')(x)


def Average_pooling(x, pool_size=(2, 2), stride=2, padding='same'):
    return AveragePooling2D(pool_size=pool_size, strides=stride, padding=padding)(x)


def Batch_Normalization(x, scope):
    return tf.keras.layers.BatchNormalization(name=scope)(x)


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Fully_connected(x, units=model_para["class_num"], layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.keras.layers.Dense(use_bias=False, units=units)(x)


class SE_ResNeXt():
    def __init__(self, x):
        self.model = self.Build_SEnet(x)

    # 第一层卷积
    def first_layer(self, x, scope):

        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=(3, 3), stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            x = Relu(x)
            return x

    # 卷积层
    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=depth, kernel=(1, 1), stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=(3, 3), stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, scope=scope+'_batch2')
            x = Relu(x)
            return x

    # 卷积层
    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=(1, 1), stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, scope=scope+'_batch1')
            # x = Relu(x)
            return x

    # 分层卷积
    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()

            for i in range(cardinality):
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    # SE层
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):

        for i in range(res_block):

            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
                channel = input_dim // 2

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True:
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x

    def Build_SEnet(self, input_x):

        input_x = self.first_layer(input_x, scope='first_layer')
        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')

        x = Global_Average_Pooling(x)
        x = Flatten()(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x


def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag='test_loss', simple_value=test_loss),
               tf.compat.v1.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


if __name__ == '__main__':

    # (train_x, train_y), (test_x, test_y) = creat_dataSet("data.mat", model_para=model_para)

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    class_num = 10
    image_size = 32
    img_channels = 3
    momentum = 0.9
    weight_decay = 0.0005
    init_learning_rate = 1.0
    total_epochs = 50
    iteration = 10
    batch_size = 32

    x = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None, class_num])

    learning_rate = tf.keras.backend.placeholder(dtype=tf.float32, name='learning_rate')

    logits = SE_ResNeXt(x).model

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    l2_loss = tf.math.add_n([tf.nn.l2_loss(var) for var in tf.compat.v1.trainable_variables()])

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)

    train = optimizer.minimize(cost + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    with tf.compat.v1.Session() as sess:

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

        summary_writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)

        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch % 30 == 0:
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):

                if pre_index + batch_size < 50000:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                }

                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

            train_loss /= iteration
            train_acc /= iteration

            train_summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='train_loss', simple_value=train_loss),
                       tf.compat.v1.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open('logs.txt', 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')
