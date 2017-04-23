## Inverse Compositional Spatial Transformer Networks
Chen-Hsuan Lin and Simon Lucey  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017  

Paper: https://www.andrew.cmu.edu/user/chenhsul/paper/CVPR2017.pdf  
arXiv preprint: https://arxiv.org/abs/1612.03897

<p align="center"><img src="https://www.andrew.cmu.edu/user/chenhsul/images/ICSTN2.png" width=600 height=250></p>

We provide the Python/TensorFlow code for the perturbed MNIST classification experiments.  
If you find our code useful for your research, please cite
```
@article{lin2017icstn,
  title={Inverse Compositional Spatial Transformer Networks},
  author={Lin, Chen-Hsuan and Lucey, Simon},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2017}
}
```

--------------------------------------

### Prerequisites  
You would need to have SciPy and TensorFlow (r0.10+) installed. Please refer to the TensorFlow documentation (https://www.tensorflow.org/) for instructions on installation/configuration.  
**[NEW]** The main branch is now compatible with the TensorFlow r1.0 API. If you have an earlier version of TensorFlow, please switch to the `r0.10+` branch after cloning.  

### Running the code  
The code is compatible with both Python2.7 (`python`) and Python3 (`python3`).  
The training code can be executed via the command
```
python train.py <TYPE> [--group GROUP] [--model MODEL] [--recurN RECURN] [--lr LR] [--lrST LRST] [--batchSize BATCHSIZE] [--maxIter MAXITER] [--warpType WARPTYPE] [--resume RESUME] [--gpu GPU]
```
`<TYPE>` should be one of the following:  
1. `CNN` - standard convolutional neural network  
2. `STN` - Spatial Transformer Network (STN)  
3. `cSTN` - compositional Spatial Transformer Network (c-STN)  
4. `ICSTN` - Inverse Compositional Spatial Transformer Network (IC-STN)  

The list of optional arguments can be found by executing `python train.py --help`.  
If no optional arguments are given, the default settings are set to be the same as described in the paper.  

When the code is run for the first time, the MNIST dataset will be automatically downloaded and preprocessed.  
The models (checkpoints) are saved in the automatically created directory `model_GROUP`; TensorFlow summaries are saved in `summary_GROUP`.

### Visualizing the results  
We've included code to visualize the training over TensorBoard. To execute, run
```
tensorboard --logdir=summary_GROUP --port=6006
```
Please refer to the TensorFlow documentation for detailed instructions.

We provide three types of data visualization:  
1. **EVENTS**: training/test error over iterations  
2. **IMAGES**: alignment results over learned spatial transformations (on test samples)  
3. **GRAPH**: network architecture

--------------------------------------

Please contact me (chenhsul@andrew.cmu.edu) if you have any questions!


