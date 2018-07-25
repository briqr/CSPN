#Creates the saliency raw maps from DCSM CNN, you can download the library and prototxt from https://github.com/shimoda-uec/dcsm, and the trained model from 
#http://mm.cs.uec.ac.jp/shimoda-k/models/mp512_iter_20000.caffemodel
import sys
sys.path.append('/home/briq/libs/dcsm_saliency/caffe/python')
import os
import numpy as np
import scipy.ndimage as nd
import caffe
import scipy.io as sio

cm = './mp512_iter_20000.caffemodel'
proto='./gdep.prototxt'
caffe.set_mode_gpu()


bsize=512
mean = np.zeros((3,int(bsize),int(bsize)))
mean[0,:,:]=104.00699
mean[1,:,:]=116.66877
mean[2,:,:]=122.67892
channel_swap = [2,1,0]
center_only=False
input_scale=None
image_dims=[bsize,bsize]
ims=(int(bsize),int(bsize))
raw_scale=255.0
data = caffe.Classifier(proto,cm, image_dims=image_dims,mean=mean,
            input_scale=input_scale,
             raw_scale=raw_scale,
            channel_swap=channel_swap)
    

size = 41
orig_img_path = '/media/datasets/VOC2012/JPEGImages/' # the path to your VOC pascal input images
img_list_path = 'input_list.txt'
save_path= '/media/VOC/saliency/raw_maps/'
with open(img_list_path) as f:
    content = f.readlines()
f.close()
content = [x.strip() for x in content]

for line in content:
    img_name = line.strip().split('.')[0]
    img_full_name = orig_img_path + img_name + '.jpg'
    im = [caffe.io.load_image(img_full_name)]
    im2=[caffe.io.resize_image(im[0], ims)]
    im3 = np.zeros((1,ims[0],ims[1],im2[0].shape[2]),dtype=np.float32)
    im3[0]=im2[0]
    caffe_in = np.zeros(np.array(im3.shape)[[0, 3, 1, 2]],dtype=np.float32)
    caffe_in[0]=data.transformer.preprocess('data', im3[0])
    out = data.forward_all(**{'data': caffe_in})
    map=data.blobs['dcsmn'].data
    id=data.blobs['sortid'].data

    for i in range(map.shape[0]):
        sn=save_path+img_name+'_'+str(int(id[0,i,0,0])+1)+'.mat'
        sio.savemat(sn, {'data':map[i,0,:,:]}) 
        
