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
        if n != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference would be the same shape as any input
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # layer output would be an averaged scalar loss
        top[0].reshape(1)

    def forward(self, bottom, top):
        features=bottom[0].data
        label=bottom[1].data
        
        n,c = features.shape[:2]
        if n % 2 != 0:
            raise Exception("batch size must be even number!")
        l2normal = np.linalg.norm(features, axis=1, keepdims=True)
        #print features.shape, l2normal.shape
        #raw_input()
        #features /= l2normal
        paires_num = (n >> 1)
        loss = float(0.0)
        #print features[0][0]
        for i in range(0, paires_num):
            f1normal = l2normal[i]
            f2normal = l2normal[paires_num + i]
            normal = f1normal * f2normal
            f1 = features[i]# / normal
            f2 = features[paires_num + i]# / normal
            inner_product = np.abs(np.sum(features[i] * features[paires_num + i].T)) / normal
            #print label[i], label[paires_num + i]
            if normal == 0:
               normal = 1
            if label[i] == label[paires_num + i]:
                loss_item = np.sqrt(1 - inner_product**2)
                loss += loss_item
                self.diff[i] = (- f2 * f1normal + f1 ** 2 / f1normal * f2) / f1normal * inner_product
                self.diff[paires_num + i] = (- f1 * f2normal + f2 ** 2 / f1normal * f1) / f2normal * inner_product
                if np.abs(loss_item) > 0:				
                   self.diff[i] /= loss_item			
                   self.diff[paires_num + i] /= loss_item
                print 'same label loss: ', inner_product
            else:
                loss_item = inner_product
                loss += loss_item
                self.diff[i] = (f2 * f1normal - f1 ** 2 / f1normal * f2)  / f1normal
                self.diff[paires_num + i] = (f1 * f2normal - f2 ** 2 / f1normal * f1) / f1normal 
                if np.abs(loss_item) > 0:				
                   self.diff[i] /= loss_item			
                   self.diff[paires_num + i] /= loss_item			
                print 'different label loss: ', inner_product		
        top[0].data[...] = np.sum(loss) / n
        #index = np.where(self.diff == 0.0)[0]
        #self.diff[index] = 1e-10
        #print loss
        if np.isnan(top[0].data):
           raise Exception("loss is non!")

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...]=self.diff
