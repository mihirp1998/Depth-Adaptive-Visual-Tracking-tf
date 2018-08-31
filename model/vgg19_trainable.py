import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]
from metrics import flopsometer

class Vgg19:


    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout



        # builds the nets with all the flops
    def build(self, rgb, train_mode=None):

        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE):

            
            self.conv1_1,flops1_1 = self.conv_layer(rgb_scaled, 3, 64, "conv1_1")
            self.conv1_2,flops1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
            self.flops1 = flops1_1 + flops1_2
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')








            self.conv2_1,flops2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2,flops2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.flops2 = flops2_1 + flops2_2 + self.flops1
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')





            self.conv3_1,flops3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2,flops3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3,flops3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4,flops3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.flops3 = flops3_1 + flops3_2 + flops3_3 + flops3_4 + self.flops2
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')





            self.conv4_1,flops4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2,flops4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3,flops4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4,flops4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')
            self.flops4 = flops4_1 + flops4_2 + flops4_3 + flops4_4 + self.flops3


            self.conv5_1,flops5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
            self.conv5_2,flops5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3,flops5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
            self.conv5_4,flops5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
            self.pool5 = self.max_pool(self.conv5_4, 'pool5')
            self.flops5 = flops5_1 + flops5_2 + flops5_3 + flops5_4 + self.flops4
            # cumuliative flops



    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            filter_size = 3
            out_channels
            # conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv,flops = flopsometer.conv2d(bottom, out_channels, 3, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu,flops



    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases



    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

