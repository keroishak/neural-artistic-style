import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
from Vgg19 import Vgg19
import scipy.misc
import scipy.io

class Stylize:
    def __init__(self,contentImgPath,styleImgPath):
        self.MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        self.OutputPath="output"
        self.WeightsFilePath="imagenet-vgg-verydeep-19.mat"
        self.CheckPoint=100
        self.Iterations=5000
        self.Width=800
        self.Height=600
        self.contentImage=self.ReadImage(contentImgPath)
        self.styleImage=self.ReadImage(styleImgPath)
        self.Style_Layers = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2),
        ];
        self.Content_Layers=[('conv4_2', 1.0)];
        self.vgg=Vgg19(self.WeightsFilePath,self.Width,self.Height)

    def ComputeContentLoss(self,p,x):
        M = p.shape[1]*p.shape[2]
        N = p.shape[3]
        return (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))
        #return 0.5 * tf.reduce_sum(tf.pow((x - p),2))
        #M = p.shape[1] * p.shape[2]
        #return (1. / (4. * N * M)) * tf.reduce_sum(tf.pow(g - p, 2))

    def ComputeStyleLoss(self,a, x):
        def _gramMatrix(F, M , N):
            Ft = tf.reshape(F, (M, N))
            return tf.matmul(tf.transpose(Ft), Ft)
        #E = [_style_loss(sess.run(graph[layer_name]), graph[layer_name]) for layer_name, _ in self.STYLE_LAYERS],
        #W = [w for _, w in self.STYLE_LAYERS],
        #W=tf.convert_to_tensor(W)
        M = a.shape[1]*a.shape[2]
        N = a.shape[3]
        A = _gramMatrix(a, M, N )
        G = _gramMatrix(x, M, N )
        return (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))

    def ReadImage(self,path):
        image = PIL.Image.open(path)
        image = image.resize((self.Width, self.Height), PIL.Image.ANTIALIAS)
        return np.asarray(image)- self.MEAN_VALUES

    def SaveImage(self,path, image):
        # Output should add back the mean.
        image = image + self.MEAN_VALUES
        # Get rid of the first useless dimension, what remains is the image.
        image = image[0]
        image = np.clip(image, 0, 255).astype('uint8')
        scipy.misc.imsave(path, image)

    def TransferStyle(self,Alpha,Beta):
        graph = self.vgg.RunConvLayers()
        print graph
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run([graph['input'].assign(self.contentImage)])
        contentLoss = sum(map(lambda l: l[1]*self.ComputeContentLoss(sess.run(graph[l[0]]) ,  graph[l[0]]), self.Content_Layers))

        sess.run([graph['input'].assign(self.styleImage)])
        styleLoss = sum(map(lambda l: l[1]*self.ComputeStyleLoss(sess.run(graph[l[0]]) , graph[l[0]]), self.Style_Layers))

        totalLoss=Alpha*contentLoss + Beta*styleLoss

        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(totalLoss)
        sess.run(tf.global_variables_initializer())

        noiseImg = np.random.uniform(-20, 20, (1, self.Height, self.Width, 3)).astype('float32')
        initialImage = 0.7* noiseImg + 0.3 * self.contentImage
        sess.run(graph['input'].assign(initialImage))
        for it in range(self.Iterations):
            sess.run(train_step)
            if it % self.CheckPoint == 0:
                synthesized_image = sess.run(graph['input'])
                # Print every 100 iteration.
                print('Iteration %d' % (it))
                print('cost: ', sess.run(totalLoss))
                self.SaveImage(self.OutputPath+"/generated %s .jpg"%(it), synthesized_image)
if __name__ == '__main__':
    style= Stylize("input/Tuebingen.jpg","input/36.jpg")
    style.TransferStyle(1,500)
