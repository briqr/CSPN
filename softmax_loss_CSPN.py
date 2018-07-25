import sys
sys.path.insert(0,'/home/briq/libs/caffe/python')
import caffe
import numpy as np

class ProjectionSoftmax(caffe.Layer):

    def softmax(self, scores):
        probs = np.zeros(scores.shape)
        maxScore = np.max(scores, axis=0)
        scores -= maxScore
        scores = np.exp(scores)
        sum_scores =  np.sum(scores, axis=0)
        probs = scores / sum_scores
        return probs


    def setup(self, bottom, top):
        self.normalization = bottom[0].num*bottom[0].shape[2]*bottom[0].shape[3]


    def reshape(self, bottom, top):
        if bottom[0].num != bottom[1].num:
            raise Exception("The true label dimension must be equal to the output dimension")
        top[0].reshape(1)

    


    def forward(self, bottom, top):
        self.projected_probs = np.zeros(bottom[1].data.shape)
        for i in range(bottom[0].num):
            self.projected_probs[i] = self.softmax(bottom[1].data[i].copy())
        
        accum_loss = np.sum(-bottom[0].data*(np.log(self.projected_probs )))

        top[0].data[...] = accum_loss / self.normalization

            


    def backward(self, top, propagate_down, bottom):
        is_argmax = False
        if(not is_argmax): # softmax
            bottom[0].diff[...] = -(self.projected_probs[...]-bottom[0].data[...])
        else: # hard argmax
            max_label = np.argmax(bottom[1].data, axis=1)
            bottom[0].diff[...] = -(self.projected_probs[...]-bottom[0].data[...]) 
            bottom[0].diff[:,max_label,:,:] -= 1
        bottom[0].diff[...] /= self.normalization
