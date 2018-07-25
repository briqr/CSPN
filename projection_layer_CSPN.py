# the simplex projection algorithm implemented as a layer, while using the saliency maps to obtain object size estimates
import sys
sys.path.insert(0,'/home/briq/libs/caffe/python')
import caffe
import random
import numpy as np
import scipy.misc
import imageio
import cv2
import scipy.ndimage as nd
import os.path
import scipy.io as sio
class SimplexProjectionLayer(caffe.Layer):


    saliency_path = '/media/VOC/saliency/thresholded_saliency_images/'
    input_list_path = '/home/briq/libs/CSPN/training/input_list.txt'

    def simplexProjectionLinear(self, data_ind, class_ind, V_im, nu):
        if(nu<1):
            return V_im
        
        heatmap_size = V_im.shape[0]*V_im.shape[1]
        theta = np.sum(V_im)
        if(theta ==nu): # the size constrain is already satisfied
            return V_im
        if(theta < nu):
            pi = V_im+(nu-theta)/heatmap_size
            return pi

        V = V_im.flatten() 
        s = 0.0
        p = 0.0
        U=V

        while(len(U) > 0):
            k = random.randint(0, len(U)-1)
            uk = U[k]
            UG = U[U>=uk]
            delta_p = len(UG)
            delta_s = np.sum(UG)
            if ((s+delta_s)-(p+delta_p)*uk<nu):
                s = s+delta_s
                p = p+delta_p
                U = U[U<uk]
            else:
                U = UG
                U = np.delete(U, np.where(U==uk))
        if(p<0.000001):
            raise ValueError('rho is too small, apparently something went wrong in the CNN')  # happens when nu<1 or V_im=infinity for example
        theta = (s-nu)/p
        pi = V_im-theta
        return pi

    def setup(self, bottom, top):
        self.num_labels = bottom[0].shape[1]
        with open(self.input_list_path) as fp:  
            self.images = fp.readlines()

        random.seed()
        

        

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
    
        for i in range(bottom[0].num):
            im_id = int(bottom[2].data[i])
            im_name = self.images[im_id].split(' ')[0].split('.')[0]
            top[0].data[i] = bottom[0].data[i]
            
            saliency_name = self.saliency_path+im_name+'.mat'
            if (not os.path.isfile(saliency_name)):
                continue
            saliency_im = sio.loadmat(saliency_name, squeeze_me=True)['data']
            for c in range(self.num_labels):
                if(c==0):
                    continue
                if(bottom[1].data[i,0,0,c]>0.5): # the label is there
                    instance = bottom[0].data[i][c]
                    nu = np.sum(saliency_im==c)
                    if(nu>1):
                        instance = bottom[0].data[i][c]
                        top[0].data[i][c]= self.simplexProjectionLinear(i, c, instance, nu)
                
            

    def backward(self, top, propagate_down, bottom):
        pass