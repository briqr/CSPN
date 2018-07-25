# Convolutional Simplex Projection Network (CSPN)
In this work we propose an optimization approach for solving weakly supervised semantic segmentation with deep Convolutional Neural Networks(CNN). 
The method introduces a novel layer which applies simplex projection on the output of a neural network in order to satisfy 
object size constraints. Although we demonstrate the performance of the approach in a semantic segmentation application, 
the method is generic and can be integrated in any given network architecture. The goal of intruducing such a layer is to
reduce the set of feasible solutions to only those that satisfy some given constraints. 
This implementation builds on [SEC](https://github.com/kolesman/SEC).
Further details on the approach can be found in the paper. This paper has been accepted in BMVC 2018. 

<img src="https://github.com/briqr/CSPN/blob/master/approach.PNG" alt="apprach" width="50%">

## Instructions:

In order to use our algorithm, you need to train a modified version of SEC CNN. But first we need some preprocessing steps.


 To obtain object size estimates, we rely on an external module for which you need to download the trained model.
1. download [DCSM library](https://github.com/shimoda-uec/dcsm), and [DCSM trained model](http://mm.cs.uec.ac.jp/shimoda-k/models/mp512_iter_20000.caffemodel). Place the files create_saliency_raw.py and create_saliency_images.py in DCSM folder. 

2. To create the saliency maps, run the file `create_saliency_raw.py` after upadting the paths to match your machine.

  These are raw class-specific saliency maps, we merge them into a single image for each input image. To do that, run:

3. `create_saliency_images.py`. 

  In this file, you need to specify a threshold value above which pixels count as salient areas, and background otherwise. In our experiments the threshold value **0.125** performed best. Note that mutliple classes could be above this threshold, so to prevent a pixel from being assigned to more than one class concurrently, we take the class with the maximum score.
  
4. download [SEC](https://github.com/kolesman/SEC) and replace their train.prototxt with our train.prototxt. 

5. You may now start training your segmentation CNN by typing in the command:

`~/libs/caffe/build/tools/caffe train --solver solver.prototxt -weights vgg16_20M.caffemodel`

  (Make sure that Caffe Python layers (*_CSPN.py as well as SEC layers) are added to your system path so that Caffe can detect them.)

6. After training is finished, you can run the code in `evaluate_VOC_val.py` to evaluate the trained CNN on Pascal VOC validation dataset.

7. Alternatively, you can download the [trained model](https://1drv.ms/u/s!AkaKZSdGrfMlhC-hsevrwVm1fdt1).  

  The optimal performance when integrating the new approach is reached at an early stage (iteration 3000), whereas in the baseline approach, a larger number of iterations was required. This fast convergence can be attributed to the fact that the space of feasible solutions is reduced only to those that are within the constraint set.

<img src="https://github.com/briqr/CSPN/blob/master/results.PNG" width="50%" alt="results">

For remarks or questions, please drop me an email (briq@iai.uni-bonn.de).

If you use our approach or rely on it, please cite our paper:


```@ARTICLE{CSPN2018,
   author = {{Briq}, R. and {Moeller}, M. and {Gall}, J.},
    title = "{Convolutional Simplex Projection Network (CSPN) for Weakly Supervised Semantic Segmentation}",
    journal = {arXiv preprint arXiv:1807.09169},
    year = {2018}
}
```

