import tensorflow as tf

from cifar10 import *

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 10

# 模型函数
import myModel.myVGGNet16 as my_VggNet16
import myModel.RESNet_0 as ResNet
import myModel.myAlexNet as AlexNet
# import myModel.test_model as testmodel
import myLayer.test_deconv_model as testmodel

# 结果目录  >>>>> + 项目名 >>>>>>>>>>
Path_all = r'E:\DATA_ALL\Result'
Path_all = Path_all + r'\pypy2_SENet'

# 数据目录
Data_all = r'E:\DATA_ALL\Picture'


def creat_ds(parament_all):

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)

    bitchsize = parament_all['BitchSize']
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(10000).batch(bitchsize)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(bitchsize)

    return train_ds, test_ds


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


def layer_show(model, load_model, img, layer_id):

    # 加载模型
    load_model = load_model + '.index'
    model.load_weights(load_model)

    img_pre = model.showNet(x=img, layer_id=layer_id)

    return img_pre


def Norm_1(img):
    img[:, :, 0] = (img[:, :, 0] - np.min(img[:, :, 0])) / (np.max(img[:, :, 0]) - np.min(img[:, :, 0]))
    img[:, :, 1] = (img[:, :, 1] - np.min(img[:, :, 1])) / (np.max(img[:, :, 1]) - np.min(img[:, :, 1]))
    img[:, :, 2] = (img[:, :, 2] - np.min(img[:, :, 2])) / (np.max(img[:, :, 2]) - np.min(img[:, :, 2]))

    return img


def arg_maxpool(pre_maxpool_x, img, strides=(2, 2)):
    img = img[0, :, :, :]
    pre_maxpool_x = pre_maxpool_x[0, :, :, :]
    w, h, f = tf.shape(img)
    W = w * strides[0]
    H = h * strides[1]

    Img = np.zeros((W, H, f))
    for chan_i in range(f):
        for i in range(w):
            for j in range(h):
                ind_w, ind_h = i * strides[0], j*strides[1]

                data2_2 = pre_maxpool_x[ind_w:ind_w+2, ind_h:ind_h+2, chan_i]
                data1_4 = np.reshape(np.array(data2_2), [1, -1])
                max_ind = np.argmax(data1_4)
                x_x = np.zeros((1, 4))
                x_x[0, max_ind] = 1

                Img[ind_w, ind_h, chan_i] = img[i, j, chan_i]*x_x[0, 0]
                Img[ind_w, ind_h+1, chan_i] = img[i, j, chan_i]*x_x[0, 1]
                Img[ind_w+1, ind_h, chan_i] = img[i, j, chan_i]*x_x[0, 2]
                Img[ind_w+1, ind_h+1, chan_i] = img[i, j, chan_i]*x_x[0, 3]

    return Img


def arg_conv(img, kearn, out_shape, strides=(1, 1)):
    Img = tf.nn.conv2d_transpose(img, kearn, output_shape=out_shape, strides=strides, padding='SAME')
    Img = tf.nn.relu(Img)

    return Img


def conv(img, kearn, strides=(2, 2)):
    Img = tf.nn.conv2d(img, kearn, strides=strides, padding='SAME')

    return Img


def arg_maxpooling(img, max_ind, out_shape=(1, 32, 32, 8)):
    _, in_W, in_H, in_C = tf.shape(img)
    Img_L = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    Img = np.zeros((1, Img_L))

    for c in range(in_C):
        for x in range(in_H):
            for y in range(in_W):
                ind = max_ind[0, y, x, c]
                Img[0, ind] = img[0, y, x, c]

    return tf.reshape(Img, out_shape)


if __name__ == "__main__":

    save_model_path = Path_all + '/保存AlexNet模型/Model1/weights.ckpt'
    save_model_acc = Path_all + '/保存AlexNet模型/Model1/ACC.npy'
    parament_all = {'Epochs': 5, 'BitchSize': 32, 'accShow': True,
                    'nClass': 10, 'input_shape': (32, 32, 3)}

    train_Run = 1

    if train_Run:

        # 模型初始化
        # Model_c = ResNet.CNNLee([2, 2, 1], num_classes=parament_all['nClass'])
        Model_c = AlexNet.AlexNet(parament_all['input_shape'], parament_all['nClass'])
        # Model_c = testmodel.test_model(parament_all['input_shape'], parament_all['nClass'])

        train_ds, test_ds = creat_ds(parament_all)
        model_run(Model_c, train_ds, test_ds, save_model_path, save_model_acc, parament_all)

    else:

        # ###################ACC结果图显示
        # result_data = np.load(save_model_acc)
        # plt_acc(result_data)

        # 加载数据
        train_x, train_y, test_x, test_y = prepare_data()
        train_x, test_x = color_preprocessing(train_x, test_x)  # Z-分数
        img = test_x[50, :, :, :]

        img_show = Norm_1(img)  # 归一化

        plt.figure(1, (8, 10))
        plt.imshow(img_show)
        plt.title('测试图片')
        plt.axis('off')
        plt.show()

        # 加载模型
        load_model = save_model_path + '.index'

        # 模型初始化
        # Model_c = testmodel.test_model(parament_all['input_shape'], parament_all['nClass'])
        Model_c = AlexNet.AlexNet(parament_all['input_shape'], parament_all['nClass'])

        Model_c.load_weights(load_model)

        img2 = img[np.newaxis, :, :, :]
        out_y = Model_c(img2, training=False)
        x1, x2, x3, p_x1, p_x2, p_x3 = Model_c.showNet(img2)

        pool_x = arg_maxpool(p_x1, x1, strides=(2, 2))

        Conv1 = Model_c.get_layer('conv1')
        Conv1 = Conv1.kernel
        pool_x = pool_x[np.newaxis, :, :, :]

        Img = arg_conv(pool_x, Conv1, out_shape=(1, 32, 32, 3), strides=(1, 1, 1, 1))
        Img = Img[0, :, :, :]

        Img = np.mean(Img, 2)

        plt.figure(2, (8, 10))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(Img[0, :, :, i])
            plt.axis('off')
        plt.show()


##################################################################################

        # 加载模型
        load_model = save_model_path + '.index'
        # Model_s = ResNet.CNNLee([2, 2, 1], num_classes=parament_all['nClass'])
        Model_c = AlexNet.AlexNet(parament_all['input_shape'], parament_all['nClass'])
        Model_c.load_weights(load_model)

        img2 = img[np.newaxis, :, :, :]
        out_y = Model_c(img2)
        pre_maxpool_x, rel_conv1 = Model_c.showNet(img2, layer_id='Conv1')

        pre_maxpool_y = arg_maxpool(pre_maxpool_x, rel_conv1, strides=(2, 2))

        plt.figure(1, (8, 8))
        for i in range(np.size(rel_conv1, 3)):
            plt.subplot(2, 4, i+1)
            plt.imshow(pre_maxpool_y[:, :, i], cmap='hot')
        plt.title('反池化')
        plt.show()

        plt.figure(2, (8, 8))
        for i in range(np.size(rel_conv1, 3)):
            plt.subplot(2, 4, i+1)
            plt.imshow(pre_maxpool_x[0, :, :, i], cmap='hot')
        plt.title('池化前')
        plt.show()

        Conv1 = Model_c.get_layer('conv1')
        Conv1 = Conv1.kernel
        pre_maxpool_y = pre_maxpool_y[np.newaxis, :, :, :]

        Img = arg_conv(pre_maxpool_y, Conv1, out_shape=(1, 32, 32, 3), strides=(1, 1, 1, 1))

        plt.figure(3, (8, 8))
        for i in range(np.size(Img, 3)):
            plt.subplot(1, 3, i+1)
            plt.imshow(Img[0, :, :, i], cmap='hot')
        plt.show()

        k = Conv1.kernel[:, :, :, 1]
        rel = Model_c(img2, training=False)
        a = np.argmax(rel)

        layer_id = 'Conv1'
        # net_layer = layer_show(Model_s, load_model, img, layer_id)
