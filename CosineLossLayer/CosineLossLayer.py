import caffe
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

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
        #features /= l2normal
        paires_num = 0
        loss = float(0.0)
        #print features
        for i in range(0, paires_num):
            while (paires_num == i):
                paires_num = random.randint(0, n-1)            
            f1normal = l2normal[i]
            f2normal = l2normal[paires_num]
            normal = f1normal * f2normal
            if normal < 1e-6:
                normal = 1.0
            f1 = features[i]
            f2 = features[paires_num]
            inner_product = np.sum(features[i] * features[paires_num].T)
            #print np.sum(inner_product)
            #raw_input()
            
            if label[i] == label[paires_num]:
                loss += 1 - inner_product / normal
                self.diff[i] = - f2 / normal
                self.diff[paires_num] = - f1 / normal
            else:
                loss += inner_product / normal
                self.diff[i] = f2 / normal
                self.diff[paires_num] = f1 / normal           
        top[0].data[...] = np.sum(loss) / n
        #print self.diff
        if np.isnan(top[0].data):
                exit()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
