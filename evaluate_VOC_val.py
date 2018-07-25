# You can use this code to evaluate the trained model of CSPN on VOC validation data, adapted from SEC

import numpy as np
import pylab

import scipy.ndimage as nd


import imageio
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

import krahenbuhl2013
import sys
sys.path.insert(0,'/home/briq/libs/caffe/python')
import caffe
import scipy

caffe.set_device(0)
caffe.set_mode_gpu()



voc_classes = [ 'background',
                'aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike',
               'person',
               'pottedplant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor',
               ]
max_label = 20

mean_pixel = np.array([104.0, 117.0, 123.0])

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)



def preprocess(image, size, mean_pixel=mean_pixel):

    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return image


def predict_mask(image_file, net, smooth=True):

    im = pylab.imread(image_file)

    net.blobs['images'].data[0] = preprocess(im, 321)
    net.forward()

    scores = np.transpose(net.blobs['fc8-SEC'].data[0], [1, 2, 0])
    d1, d2 = float(im.shape[0]), float(im.shape[1])
    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
    probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result

def evaluate(res, gt_img):
    intersect_gt_res = np.sum( (res == gt_img) & (res!=0) & (gt_img!=0) )
    union_gt_res = np.sum( (res!=0) | (gt_img!=0) )
    acc = float(intersect_gt_res) / union_gt_res
    return acc



model =  '/home/briq/libs/CSPN/training/models/model_iter_3000.caffemodel' 
draw = False
smoothing = True


if __name__ == "__main__":

    num_classes = len(voc_classes)
    gt_path = '/media/datasets/VOC2012/SegmentationClassAug/'
    
    orig_img_path = '/media/datasets/VOC2012/JPEGImages/'
    img_list_path = '/home/briq/libs/CSPN/list/val_id.txt'

    with open(img_list_path) as f:
        content = f.readlines()
    f.close()
    content = [x.strip() for x in content]

    num_ims = 0
    
    cspn_net = caffe.Net('deploy.prototxt', model, caffe.TEST)

    for line in content:
        img_name = line.strip()


        gt_name = gt_path + img_name
        gt_name = gt_name + '.png'

        gt_img = imageio.imread(gt_name)


        orig_img_name = orig_img_path + img_name
        orig_img_name = orig_img_name + '.jpg'
        res =  predict_mask(orig_img_name, cspn_net, smooth=smoothing) 
        
        num_ims += 1
        if(num_ims%100==0):
            print '-----------------im:{}---------------------\n'.format(num_ims)

    
        acc = evaluate(res, gt_img)

        print img_name, str(num_ims), "{}%\n".format(acc*100)
        
        if draw:
            fig = plt.figure()
            ax = fig.add_subplot('221')
            ax.imshow(pylab.imread(orig_img_name))
            plt.title('image')
            
            ax = fig.add_subplot('222')
            ax.matshow(gt_img, vmin=0, vmax=21, cmap=my_cmap)
            plt.title('GT')
            
            ax = fig.add_subplot('223')
            ax.matshow(res, vmin=0, vmax=21, cmap=my_cmap)
            plt.title('CSPN')

            
            plt.show()


 
