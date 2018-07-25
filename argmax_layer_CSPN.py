import sys
sys.path.insert(0,'/home/briq/libs/caffe/python')
import caffe
import random
import numpy as np
import scipy.misc


class ArgmaxLayer(caffe.Layer):
   

    def setup(self, bottom, top):
        pass
      
    
    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].shape[2], bottom[0].shape[3])



    def forward(self, bottom, top):
        top[0].data[...] = np.argmax(bottom[0].data[...], axis=1) 


    def backward(self, top, propagate_down, bottom):
        propagate_down = False
        pass
