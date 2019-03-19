## Inverse Compositional Spatial Transformer Networks
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/)
and [Simon Lucey](http://www.simonlucey.com/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017 (**oral presentation**)  

Project page: https://chenhsuanlin.bitbucket.io/inverse-compositional-STN  
Paper: https://chenhsuanlin.bitbucket.io/inverse-compositional-STN/paper.pdf  
Poster: https://chenhsuanlin.bitbucket.io/inverse-compositional-STN/poster.pdf   
arXiv preprint: https://arxiv.org/abs/1612.03897

<p align="center"><img src="https://www.andrew.cmu.edu/user/chenhsul/images/ICSTN2.png" width=600 height=250></p>

We provide TensorFlow code for the following experiments:
- MNIST classification
- traffic sign classification

**[NEW!]** The PyTorch implementation of the MNIST experiment is now up!  

--------------------------------------

## TensorFlow

### Prerequisites  
This code is developed with Python3 (`python3`) but it is also compatible with Python2.7 (`python`). TensorFlow r1.0+ is required. The dependencies can install by running
```
pip3 install --upgrade numpy scipy termcolor matplotlib tensorflow-gpu
```
If you're using Python2.7, use `pip2` instead; if you don't have sudo access, add the `--user` flag.  

### Running the code  
The training code can be executed via the command
```
python3 train.py <netType> [(options)]
```
`<netType>` should be one of the following:  
1. `CNN` - standard convolutional neural network  
2. `STN` - Spatial Transformer Network (STN)  
3. `IC-STN` - Inverse Compositional Spatial Transformer Network (IC-STN)  

The list of optional arguments can be found by executing `python3 train.py --help`.  
The default training settings in this released code is slightly different from that in the paper; it is stabler and optimizes the networks better.  

When the code is run for the first time, the datasets will be automatically downloaded and preprocessed.  
The checkpoints are saved in the automatically created directory `model_GROUP`; summaries are saved in `summary_GROUP`.

### Visualizing the results  
We've included code to visualize the training over TensorBoard. To execute, run
```
tensorboard --logdir=summary_GROUP --port=6006
```

We provide three types of data visualization:  
1. **SCALARS**: training/test error over iterations  
2. **IMAGES**: alignment results and mean/variance appearances  
3. **GRAPH**: network architecture

--------------------------------------

## PyTorch

The PyTorch version of the code is stil under active development. The training speed is currently slower than the TensorFlow version. Suggestions on improvements are welcome! :)

### Prerequisites  
This code is developed with Python3 (`python3`). It has not been tested with Python2.7 yet. PyTorch 0.2.0+ is required. Please see http://pytorch.org/ for installation instructions.  
Visdom is also required; it can be installed by running
```
pip3 install --upgrade visdom
```
If you don't have sudo access, add the `--user` flag.  

### Running the code  
First, start a Visdom server by running
```
python3 -m visdom.server -port=7000
```
The training code can be executed via the command (using the same port number)
```
python3 train.py <netType> --port=7000 [(options)]
```
`<netType>` should be one of the following:  
1. `CNN` - standard convolutional neural network  
2. `STN` - Spatial Transformer Network (STN)  
3. `IC-STN` - Inverse Compositional Spatial Transformer Network (IC-STN)  

The list of optional arguments can be found by executing `python3 train.py --help`.  
The default training settings in this released code is slightly different from that in the paper; it is stabler and optimizes the networks better.  

When the code is run for the first time, the datasets will be automatically downloaded and preprocessed.  
The checkpoints are saved in the automatically created directory `model_GROUP`; summaries are saved in `summary_GROUP`.

### Visualizing the results  
We provide three types of data visualization on Visdom:  
1. Training/test error over iterations  
2. Alignment results and mean/variance appearances  

--------------------------------------

If you find our code useful for your research, please cite
```
@inproceedings{lin2017inverse,
  title={Inverse Compositional Spatial Transformer Networks},
  author={Lin, Chen-Hsuan and Lucey, Simon},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2017}
}
```

Please contact me (chlin@cmu.edu) if you have any questions!


