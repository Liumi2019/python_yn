import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models


class AlexNet(k_models.Model):

    def __init__(self, input_shape, nclass):
        super(AlexNet, self).__init__()
        self.inputlayer = k_layers.InputLayer(input_shape=input_shape, name='inputlayer')
        self.conv1 = k_layers.Conv2D(8, (3, 3), strides=(1, 1), name='conv1', padding='same', activation='relu')
        self.conv2 = k_layers.Conv2D(16, (3, 3), strides=(1, 1), name='conv2', padding='same', activation='relu')
        self.conv3 = k_layers.Conv2D(32, (3, 3), strides=(1, 1), name='conv3', padding='same', activation='relu')
        self.conv4 = k_layers.Conv2D(64, (3, 3), strides=(1, 1), name='conv4', padding='same', activation='relu')
        self.conv5 = k_layers.Conv2D(64, (3, 3), strides=(1, 1), name='conv5', padding='same', activation='relu')

        self.maxpool1 = k_layers.MaxPool2D(name='maxpool1', strides=(2, 2))
        self.maxpool2 = k_layers.MaxPool2D(name='maxpool2', strides=(2, 2))
        self.maxpool3 = k_layers.MaxPool2D(name='maxpool3', strides=(2, 2))
        self.maxpool4 = k_layers.MaxPool2D(name='maxpool4', strides=(2, 2))
        self.maxpool5 = k_layers.MaxPool2D(name='maxpool5', strides=(2, 2))

        self.bnlze1 = k_layers.BatchNormalization(name='bn1')
        self.bnlze2 = k_layers.BatchNormalization(name='bn2')
        self.bnlze3 = k_layers.BatchNormalization(name='bn3')
        self.bnlze4 = k_layers.BatchNormalization(name='bn4')
        self.bnlze5 = k_layers.BatchNormalization(name='bn5')

        self.flt = k_layers.Flatten()
        self.fc1 = k_layers.Dense(64, name='fc1', activation='relu')
        self.fc2 = k_layers.Dense(128, name='fc2', activation='relu')
        self.fc3 = k_layers.Dense(nclass, name='outlayer', activation='softmax')

    @tf.function
    def call(self, x, training=True):

        # 第一块
        x = self.inputlayer(x)
        x = self.conv1(x, training=training)
        x = self.maxpool1(x)
        x = self.bnlze1(x, training=training)
        # 第二块
        x = self.conv2(x, training=training)
        x = self.maxpool2(x)
        x = self.bnlze2(x, training=training)
        # 第三块
        x = self.conv3(x, training=training)
        x = self.maxpool3(x)
        x = self.bnlze3(x, training=training)
        # 第四块
        x = self.conv4(x, training=training)
        x = self.maxpool4(x)
        x = self.bnlze4(x, training=training)
        # 第五块
        x = self.conv5(x, training=training)
        x = self.maxpool5(x)
        x = self.bnlze5(x, training=training)
        # 输出
        x = self.flt(x)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)

        return x

    @tf.function
    def showNet(self, x):

        training = False

        # 第一块
        x = self.inputlayer(x)
        x = self.conv1(x, training=training)
        p_x1 = x
        x = self.maxpool1(x)
        x = self.bnlze1(x, training=training)
        x1 = x
        # 第二块
        x = self.conv2(x, training=training)
        p_x2 = x
        x = self.maxpool2(x)
        x = self.bnlze2(x, training=training)
        x2 = x
        # 第三块
        x = self.conv3(x, training=training)
        p_x3 = x
        x = self.maxpool3(x)
        x = self.bnlze3(x, training=training)
        x3 = x
        # 第四块
        x = self.conv4(x, training=training)
        x = self.maxpool4(x)
        x = self.bnlze4(x, training=training)
        x4 = x
        # 第五块
        x = self.conv5(x, training=training)
        x = self.maxpool5(x)
        x = self.bnlze5(x, training=training)
        x5 = x

        return x1, x2, x3, p_x1, p_x2, p_x3


