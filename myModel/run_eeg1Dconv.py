import tensorflow as tf
import numpy as np
import os

import myModel.EEG1DConv as eegnet

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 10


def model_run(model, train_ds, test_ds, model_path, save_model_acc, parament_all):

    is_model_path = model_path + '.index'

    if os.path.exists(is_model_path):
        model.load_weights(model_path)
        print("加载的模型")
    else:
        print("新建一个模型")

    # 损失和优化定义
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # 打印训练和测试指标
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):  # 训练
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):  # 测试
        predictions = model(images, training=False)
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

        # 训练
        for images, labels in train_ds:
            train_step(images, labels)

        # 测试
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

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

    np.save(save_model_acc, result_data)

    if parament_all['accShow']:
        plt_acc(result_data)


def plt_acc(result_acc):

    data_len = np.size(result_acc, 1)

    if parament_all['accShow']:
        plt.plot(np.arange(data_len), result_acc[0, :], 'b--')
        plt.plot(np.arange(data_len), result_acc[1, :], 'b')
        plt.plot(np.arange(data_len), result_acc[2, :], 'r--')
        plt.plot(np.arange(data_len), result_acc[3, :], 'r')
        plt.legend(['训练损失', '训练准确率', '测试损失', '测试准确率'])
        plt.show()


def creat_ds(parament_all):

    train_x = tf.random.normal((100, 8, 400, 1), stddev=5.0, seed=50)
    test_x = tf.random.normal((50, 8, 400, 1), stddev=2.0, seed=40)
    train_y = tf.random.normal((100, 1, 100), seed=15)
    test_y = tf.random.normal((50, 1, 100), seed=20)

    bitchsize = parament_all['BitchSize']
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(100).batch(bitchsize)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(bitchsize)

    return train_ds, test_ds


if __name__ == "__main__":

    all_path = 'F:/SYPY/liu_matlab_EEG/result/python_result/新建模型/'

    save_model_path = all_path + '保存EEG1Dconv模型/Model1/weights.ckpt'
    save_model_acc = all_path + '保存EEG1Dconv模型/Model1/ACC.npy'

    parament_all = {'Epochs': 20, 'BitchSize': 32, 'accShow': True,
                    'nClass': (1, 100), 'input_shape': (8, 400, 1)}

    ###################################################################

    train_Run = 1

    if train_Run:

        # 模型初始化
        Model_c = eegnet.EEG1DConv(parament_all['input_shape'], parament_all['nClass'])

        train_ds, test_ds = creat_ds(parament_all)
        model_run(Model_c, train_ds, test_ds, save_model_path, save_model_acc, parament_all)
