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


def normalize(img):
    img_max, img_min = tf.reduce_max(img, 2), tf.reduce_min(img, 2)
    img2 = np.zeros((tf.shape(img)))
    img2[:, :, 0] = (img[:, :, 0] - img_min) / (img_max - img_min)
    img2[:, :, 1] = (img[:, :, 1] - img_min) / (img_max - img_min)
    img2[:, :, 2] = (img[:, :, 2] - img_min) / (img_max - img_min)

    return img2


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


# 图像基线与图像插值
def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta

    return images


# 计算梯度
def compute_gradients(model, images, target_class_idx):

    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]

    return tape.gradient(probs, images)


# 积分梯度
def integral_approximation(gradients):

    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients


@tf.function
def integrated_gradients(model, baseline, image, target_class_idx, m_steps=50, batch_size=32):

    # 1、近似积分的微变量，间隔为alpha。
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # 可按步骤收集不同的矩阵值
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):

        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))

        alpha_batch = alphas[from_:to]

        # 2、积分近似分割
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)

        # 3、计算梯度变换
        gradient_batch = compute_gradients(model=model, images=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

        # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def plot_img_attributions(model, baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):

    attributions = integrated_gradients(model=model, baseline=baseline, image=image, target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()

    return fig


# IG可视化一共三步
# 1、产生基线图像
# 2、基线图与原图间插值得到一系列积分近似变化值
# 3、求不同变化值带来的叠加和，近似为积分值

if __name__ == "__main__":

    save_model_path = 'E:/pypy2_SENet/tensorflow_learn/保存AlexNet模型/Model1/weights.ckpt'
    save_model_acc = 'E:/pypy2_SENet/tensorflow_learn/保存AlexNet模型/Model1/ACC.npy'

    parament_all = {'Epochs': 5, 'BitchSize': 32, 'accShow': True,
                    'nClass': 10, 'input_shape': (32, 32, 3)}

    train_Run = 0

    if train_Run:

        # 模型初始化
        # Model_c = ResNet.CNNLee([2, 2, 1], num_classes=parament_all['nClass'])
        Model_c = AlexNet.AlexNet(parament_all['input_shape'], parament_all['nClass'])
        # Model_c = testmodel.test_model(parament_all['input_shape'], parament_all['nClass'])

        train_ds, test_ds = creat_ds(parament_all)
        model_run(Model_c, train_ds, test_ds, save_model_path, save_model_acc, parament_all)

    else:

        # 加载模型
        Model_c = AlexNet.AlexNet(parament_all['input_shape'], parament_all['nClass'])
        load_model = save_model_path + '.index'
        Model_c.load_weights(load_model)

        # 加载数据
        train_x, train_y, test_x, test_y = prepare_data()
        train_x, test_x = color_preprocessing(train_x, test_x)  # Z-分数

        # 测试图像的索引和图片名称的索引
        img_num = 50
        # img_ind = 0  # 索引：0-9

        img = test_x[img_num, :, :, :]
        img = Norm_1(img)
        img_onehot = test_y[img_num, :]
        img_ind = tf.argmax(img_onehot)

        basd_img = tf.zeros(shape=tf.shape(img))

        # 图像显示标志
        img_show_mark = 0
        if img_show_mark:
            img_show = Norm_1(img)  # 归一化，便于显示

            plt.figure(1, (8, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(img_show)
            plt.title('测试图片')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(basd_img)
            plt.title('基线')
            plt.axis('off')
            plt.show()

        # 总的运行结果
        # _ = plot_img_attributions(model=Model_c, baseline=basd_img, image=img, target_class_idx=img_ind,
        #                           m_steps=30, cmap=plt.cm.inferno, overlay_alpha=0.4)
        #
        # plt.show()

        # ####### 分步学习 ##########

        m_steps = 50
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

        inter_images = interpolate_images(baseline=basd_img, image=img, alphas=alphas)

        fig = plt.figure(2, figsize=(8, 10))
        i = 0
        for alpha, image in zip(alphas[0::10], inter_images[0::10]):
            i += 1
            plt.subplot(1, len(alphas[0::10]), i)
            plt.title(f'alpha: {alpha:.1f}')
            plt.imshow(image)
            plt.axis('off')

        plt.tight_layout()
        # plt.show()

        path_gradients = compute_gradients(model=Model_c, images=inter_images, target_class_idx=3)
        # print(path_gradients.shape)
        j = 0
        for i in np.arange(0, 50, 5):
            j = j + 1
            plt.subplot(2, 5, j)
            img_graid = tf.reduce_max(path_gradients[i, :, :, :], 2) * 100
            # img_graid = normalize(img_graid)
            plt.imshow(img_graid)
            plt.axis('off')

        pred = Model_c(inter_images)
        pred_proba = tf.nn.softmax(pred, axis=-1)[:, 3]

        plt.figure(3, figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(alphas, pred_proba)
        ax1.set_title('Target class predicted probability over alpha')
        ax1.set_ylabel('model p(target class)')
        ax1.set_xlabel('alpha')
        ax1.set_ylim([0, 1])

        ax2 = plt.subplot(1, 2, 2)
        # Average across interpolation steps
        average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
        # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
        average_grads_norm = (average_grads - tf.math.reduce_min(average_grads)) / (
                    tf.math.reduce_max(average_grads) - tf.reduce_min(average_grads))
        ax2.plot(alphas, average_grads_norm)
        ax2.set_title('Average pixel gradients (normalized) over alpha')
        ax2.set_ylabel('Average pixel gradients')
        ax2.set_xlabel('alpha')
        ax2.set_ylim([0, 1])
        # plt.show()

        ig = integral_approximation(gradients=path_gradients)

        plt.figure(4)
        plt.imshow(ig)





