## Inverse Compositional Spatial Transformer Networks
Chen-Hsuan Lin and Simon Lucey  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017 (**oral presentation**)  

Paper: https://www.andrew.cmu.edu/user/chenhsul/paper/CVPR2017.pdf  
arXiv preprint: https://arxiv.org/abs/1612.03897

<p align="center"><img src="https://www.andrew.cmu.edu/user/chenhsul/images/ICSTN2.png" width=600 height=250></p>

We provide the Python/TensorFlow code for the perturbed MNIST classification experiments.  
If you find our code useful for your research, please cite
```
@article{lin2017inverse,
  title={Inverse Compositional Spatial Transformer Networks},
  author={Lin, Chen-Hsuan and Lucey, Simon},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2017}
}
```

--------------------------------------

### Prerequisites  
This code is developed with Python3. The following Python packages are required: 
- TensorFlow (r1.0+)
- NumPy
- SciPy
- TermColor  

You can install them by running the command line
```
pip3 install --upgrade numpy scipy termcolor tensorflow-gpu
```
If you use Python2.7, use `pip2` instead; if you don't have sudo access, add the `--user` flag.  

### Running the code  
The code is compatible with both Python3 (`python3`) and Python2.7 (`python`).  
The training code can be executed via the command
```
python3 train.py <netType> [(options)]
```
`<netType>` should be one of the following:  
1. `CNN` - standard convolutional neural network  
2. `STN` - Spatial Transformer Network (STN)  
3. `IC-STN` - Inverse Compositional Spatial Transformer Network (IC-STN)  

The list of optional arguments can be found by executing `python3 train.py --help`.  
The default settings in this code is slightly different from that in the paper; it is faster and more stable to optimize.  

When the code is run for the first time, the MNIST dataset will be automatically downloaded and preprocessed.  
The checkpoints are saved in the automatically created directory `model_GROUP`; summaries are saved in `summary_GROUP`.

### Visualizing the results  
We've included code to visualize the training over TensorBoard. To execute, run
```
tensorboard --logdir=summary_GROUP --port=6006
```

We provide three types of data visualization:  
1. **SCALARS**: training/test error over iterations  
2. **IMAGES**: alignment results over learned spatial transformations  
3. **GRAPH**: network architecture

--------------------------------------

Please contact me (chlin@cmu.edu) if you have any questions!


