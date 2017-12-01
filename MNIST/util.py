import numpy as np
import scipy.misc
import tensorflow as tf
import os
import termcolor

def mkdir(path):
	if not os.path.exists(path): os.mkdir(path)
def imread(fname):
	return scipy.misc.imread(fname)/255.0
def imsave(fname,array):
	scipy.misc.toimage(array,cmin=0.0,cmax=1.0).save(fname)

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])

# make image summary from image batch
def imageSummary(opt,image,tag,H,W):
	blockSize = opt.visBlockSize
	imageOne = tf.batch_to_space(image[:blockSize**2],crops=[[0,0],[0,0]],block_size=blockSize)
	imagePermute = tf.reshape(imageOne,[H,blockSize,W,blockSize,-1])
	imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
	imageBlocks = tf.reshape(imageTransp,[1,H*blockSize,W*blockSize,-1])
	imageBlocks = tf.cast(imageBlocks*255,tf.uint8)
	summary = tf.summary.image(tag,imageBlocks)
	return summary

# make image summary from image batch (mean/variance)
def imageSummaryMeanVar(opt,image,tag,H,W):
	imageOne = tf.batch_to_space_nd(image,crops=[[0,0],[0,0]],block_shape=[1,10])
	imagePermute = tf.reshape(imageOne,[H,1,W,10,-1])
	imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
	imageBlocks = tf.reshape(imageTransp,[1,H*1,W*10,-1])
	imageBlocks = tf.cast(imageBlocks*255,tf.uint8)
	summary = tf.summary.image(tag,imageBlocks)
	return summary

# set optimizer for different learning rates
def setOptimizer(opt,loss,lrGP,lrC):
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]
	varsC = [v for v in tf.global_variables() if "classifier" in v.name]
	gradC = tf.gradients(loss,varsC)
	optimC = tf.train.GradientDescentOptimizer(lrC).apply_gradients(zip(gradC,varsC))
	if len(varsGP)>0:
		gradGP = tf.gradients(loss,varsGP)
		optimGP = tf.train.GradientDescentOptimizer(lrGP).apply_gradients(zip(gradGP,varsGP))
		optim = tf.group(optimC,optimGP)
	else:
		optim = optimC
	return optim

# restore model
def restoreModel(opt,sess,saver,it):
	saver.restore(sess,"models_{0}/{1}_it{2}.ckpt".format(opt.group,opt.model,it,opt.warpN))
# save model
def saveModel(opt,sess,saver,it):
	saver.save(sess,"models_{0}/{1}_it{2}.ckpt".format(opt.group,opt.model,it,opt.warpN))

