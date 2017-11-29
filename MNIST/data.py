import numpy as np
import scipy.linalg
import os,time
import tensorflow as tf

import warp

# load MNIST data
def loadMNIST(fname):
	if not os.path.exists(fname):
		# download and preprocess MNIST dataset
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
		trainData,validData,testData = {},{},{}
		trainData["image"] = mnist.train.images.reshape([-1,28,28]).astype(np.float32)
		validData["image"] = mnist.validation.images.reshape([-1,28,28]).astype(np.float32)
		testData["image"] = mnist.test.images.reshape([-1,28,28]).astype(np.float32)
		trainData["label"] = mnist.train.labels.astype(np.float32)
		validData["label"] = mnist.validation.labels.astype(np.float32)
		testData["label"] = mnist.test.labels.astype(np.float32)
		os.makedirs(os.path.dirname(fname))
		np.savez(fname,train=trainData,valid=validData,test=testData)
		os.system("rm -rf MNIST_data")
	MNIST = np.load(fname)
	trainData = MNIST["train"].item()
	validData = MNIST["valid"].item()
	testData = MNIST["test"].item()
	return trainData,validData,testData

# generate training batch
def genPerturbations(opt):
	with tf.name_scope("genPerturbations"):
		X = np.tile(opt.canon4pts[:,0],[opt.batchSize,1])
		Y = np.tile(opt.canon4pts[:,1],[opt.batchSize,1])
		dX = tf.random_normal([opt.batchSize,4])*opt.pertScale \
			+tf.random_normal([opt.batchSize,1])*opt.transScale
		dY = tf.random_normal([opt.batchSize,4])*opt.pertScale \
			+tf.random_normal([opt.batchSize,1])*opt.transScale
		O = np.zeros([opt.batchSize,4],dtype=np.float32)
		I = np.ones([opt.batchSize,4],dtype=np.float32)
		# fit warp parameters to generated displacements
		if opt.warpType=="homography":
			A = tf.concat([tf.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
						   tf.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],1)
			b = tf.expand_dims(tf.concat([X+dX,Y+dY],1),-1)
			pPert = tf.matrix_solve(A,b)[:,:,0]
			pPert -= tf.to_float([[1,0,0,0,1,0,0,0]])
		else:
			if opt.warpType=="translation":
				J = np.concatenate([np.stack([I,O],axis=-1),
									np.stack([O,I],axis=-1)],axis=1)
			if opt.warpType=="similarity":
				J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
									np.stack([-Y,X,O,I],axis=-1)],axis=1)
			if opt.warpType=="affine":
				J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
									np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
			dXY = tf.expand_dims(tf.concat(1,[dX,dY]),-1)
			pPert = tf.matrix_solve_ls(J,dXY)[:,:,0]
	return pPert

# make training batch
def makeBatch(opt,data,PH):
	N = len(data["image"])
	randIdx = np.random.randint(N,size=[opt.batchSize])
	# put data in placeholders
	[image,label] = PH
	batch = {
		image: data["image"][randIdx],
		label: data["label"][randIdx],
	}
	return batch

# evaluation on validation/test sets
def evaluate(opt,sess,data,PH,prediction):
	N = len(data["image"])
	# put data in placeholders
	[image,label] = PH
	batchN = int(np.ceil(N/opt.batchSize))
	count = 0
	for b in range(batchN):
		# use some dummy data (0) as batch filler if necesaary
		if b!=batchN-1:
			realIdx = np.arange(opt.batchSize*b,opt.batchSize*(b+1))
		else:
			realIdx = np.arange(opt.batchSize*b,N)
		idx = np.zeros([opt.batchSize],dtype=int)
		idx[:len(realIdx)] = realIdx
		batch = {
			image: data["image"][idx],
			label: data["label"][idx],
		}
		pred = sess.run(prediction,feed_dict=batch)
		count += pred[:len(realIdx)].sum()
	accuracy = float(count)/N
	return accuracy
