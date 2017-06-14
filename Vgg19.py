import numpy as np
import tensorflow as tf
import scipy.io

class Vgg19:
        def __init__(self, weights_file_path,width,height):
            self.Width=width
            self.Height=height
            self.weights = scipy.io.loadmat(weights_file_path)['layers'][0]

        def ConvRelu(self, input, layer_index,layer_name):
            w = self.weights[layer_index][0][0][0][0][0]
            b = self.weights[layer_index][0][0][0][0][1]
            kernel = tf.constant(w)
            k=np.reshape(b, (b.size))
            bias = tf.constant(k)
            conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(conv + bias)

        def RunConvLayers(self):
            graph={}

            graph['input'] = tf.Variable(tf.zeros([1,self.Height,self.Width,3]), tf.float32)

            graph['conv1_1'] = self.ConvRelu(graph['input'],0,'conv1_1')
            graph['conv1_2'] = self.ConvRelu(graph['conv1_1'], 2, 'conv1_2')
            graph['poo1'] = tf.nn.max_pool(graph['conv1_2'],ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

            graph['conv2_1'] = self.ConvRelu(graph['poo1'],5,'conv2_1')
            graph['conv2_2'] = self.ConvRelu(graph['conv2_1'], 7, 'conv2_2')
            graph['pool2'] = tf.nn.avg_pool(graph['conv2_2'],ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')


            graph['conv3_1'] = self.ConvRelu(graph['pool2'],10,'conv3_1')
            graph['conv3_2'] = self.ConvRelu(graph['conv3_1'], 12, 'conv3_2')
            graph['conv3_3'] = self.ConvRelu(graph['conv3_2'], 14, 'conv3_3')
            graph['conv3_4'] = self.ConvRelu(graph['conv3_3'], 16, 'conv3_4')
            graph['pool3'] = tf.nn.avg_pool(graph['conv3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            graph['conv4_1'] = self.ConvRelu(graph['pool3'], 19,'conv4_1')
            graph['conv4_2'] = self.ConvRelu(graph['conv4_1'], 21, 'conv4_2')
            graph['conv4_3'] = self.ConvRelu(graph['conv4_2'], 32, 'conv4_3')
            graph['conv4_4'] = self.ConvRelu(graph['conv4_3'], 25, 'conv4_4')
            graph['pool4'] = tf.nn.avg_pool(graph['conv4_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

            graph['conv5_1'] = self.ConvRelu(graph['pool4'], 28,'conv5_1')
            graph['conv5_2'] = self.ConvRelu(graph['conv5_1'], 30, 'conv5_2')
            graph['conv5_3'] = self.ConvRelu(graph['conv5_2'], 32, 'conv5_3')
            graph['conv5_4'] = self.ConvRelu(graph['conv5_3'], 34, 'conv5_4')
            graph['pool5'] = tf.nn.avg_pool(graph['conv5_4'], ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
            return graph
