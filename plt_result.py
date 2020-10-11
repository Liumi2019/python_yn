import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras.models as kmodel
import tensorflow.keras.backend as K


def plot_result():
    matplotlib.rcParams['font.family'] = 'STSong'
    matplotlib.rcParams['font.size'] = 15

    result_data = np.load('LenetNoSEresult.npy')
    EPOCHS = 20
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(EPOCHS), result_data[0, :], 'b--')
    plt.plot(np.arange(EPOCHS), result_data[1, :], 'b')
    plt.plot(np.arange(EPOCHS), result_data[2, :], 'r--')
    plt.plot(np.arange(EPOCHS), result_data[3, :], 'r')
    plt.legend(['训练损失', '训练准确率', '测试损失', '测试准确率'])

    result_data = np.load('LenetSEresult.npy')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(EPOCHS), result_data[0, :], 'b--')
    plt.plot(np.arange(EPOCHS), result_data[1, :], 'b')
    plt.plot(np.arange(EPOCHS), result_data[2, :], 'r--')
    plt.plot(np.arange(EPOCHS), result_data[3, :], 'r')
    plt.legend(['训练损失', '训练准确率', '测试损失', '测试准确率'])
    plt.show()


if __name__ == '__main__':

    img = np.load('img.npy')

    plt.imshow(img)
    plt.show()

    model = tf.keras.models.load_model('保存LeNet模型/SE')

    img = img[np.newaxis, :, :, :]
    a = model.predict(img)
    pre = tf.argmax(a, 1) + 1

    plt.imshow(model.showNet(x=img, layer_id='conv1'))
    plt.show()

    layer = model.get_layer('conv1_maxpool')

    heatmap_model = kmodel.Model([model.inputs], [layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, Predictions = heatmap_model(img)
        prob = Predictions[:, np.argmax(Predictions[0])]  # 最大可能性类别的预测概率
        grads = gtape.gradient(prob, conv_output)  # 类别与卷积层的梯度 (1,14,14,512)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)  # 权重与特征层相乘，512层求和平均

    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    plt.matshow(heatmap[0], cmap='viridis')
