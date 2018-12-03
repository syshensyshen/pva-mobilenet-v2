import caffe
import scipy
import numpy as np
import random
np.seterr(divide='ignore', invalid='ignore')
'''
this implement by syshen
using cosine distance for backward
L2 normal in denominator, and Derivation of it
'''
class CosineLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check for all inputs
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute sigmoid crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match between the scores and labels
        features = bottom[0].data
        n,c = features.shape[:2]
        if n < 2:
            raise Exception("Inputs must great than two sample.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        features=bottom[0].data
        label=bottom[1].data        
        n,c = features.shape[:2]
        l2normal = np.linalg.norm(features, axis=1, keepdims=True)
        #print features.shape, l2normal.shape
        #raw_input()
        loss = float(0.0)
        paires_num = 0
        pairs = [i for i in range(n)]
        pairs = np.array(pairs, dtype=np.int32)
        random.shuffle(pairs)
        #print pairs
        pairs = pairs.reshape((-1, 2))
        h,w = pairs.shape
        for i in range(0, h>>1):
            first =  pairs[i,0]
            second = pairs[i,1]
            f1normal = l2normal[first]
            f2normal = l2normal[second]
            if f1normal == 0:
                f1normal = 1
            if f2normal == 0:
                f2normal = 1
            normal = f1normal * f2normal
            f1 = features[first]# / normal
            f2 = features[second]# / normal
            inner_product = np.abs(np.sum(features[first] * features[second].T)) / normal
            self.diff[first] = (f2 * f1normal - f1 ** 2 / f1normal * f2)  / f1normal
            self.diff[second] = (f1 * f2normal - f2 ** 2 / f1normal * f1) / f1normal
            signal = 1
            if label[first] == label[second]:
                loss += 1 - inner_product
                signal = -1
            else:
                loss += inner_product
            self.diff[first] *= signal
            self.diff[second] *= signal 	
        top[0].data[...] = np.sum(loss) / n * 2
        if np.isnan(top[0].data):
            print 'data: ', features
            print 'diff: ', self.diff
            raise Exception("loss is non!")

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
