import sys
import numpy as np
import scipy.misc
import scipy.ndimage as nd
import os.path
import scipy.io as sio




saliency_path = '/media/VOC/saliency/raw_maps/'  # the path of the raw class-specific saliency maps, created by create_saliency_raw.py
save_path = '/media/VOC/saliency/thresholded_saliency_images/' # the path where combined class-specific saliency maps will be saved after thresholding
dataset_path = 'val.txt'
size = 41 #corresponds to the dimension of fc8 in the CNN

with open(dataset_path) as fp:  
  images = fp.readlines()
for im_id in range(len(images)):
    import os

    im_name = images[im_id].split(' ')[0].split('.')[0].split('/')[2]

    saliency_ims = []
    threshold = 0.125
    bkg = np.ones((size, size))*2
    for c in range(20):
      if(c==0):
        saliency_ims.append(np.zeros((size,size)))
        continue
      
      saliency_name = saliency_path+im_name+'_' + str(c)+'.mat'
      if (not os.path.isfile(saliency_name)):
        saliency_ims.append(np.ones((size,size)))
        saliency_ims[c] *= -2 # just to make sure non occuring classes will never turn up in the argmax operation
        continue
      
      saliency_map = sio.loadmat(saliency_name, squeeze_me=True)['data'] #
 
      saliency_map = nd.zoom(saliency_map.astype('float32'), (size / float(saliency_map.shape[0]), size / float(saliency_map.shape[1]) ), order=1)
      saliency_map[saliency_map<threshold]=0
      bkg[np.where(saliency_map>=threshold)]=0 # mark the saliency pixels as non background
      saliency_ims.append(saliency_map)
      saliency_ims[0] = bkg
      total_name = save_path+im_name+'.mat' 
      total_im=np.argmax(saliency_ims, axis=0) 
      sio.savemat(total_name , {'data':total_im})
