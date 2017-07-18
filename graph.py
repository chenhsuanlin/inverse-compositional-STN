import numpy as np
import tensorflow as tf
import time
import data,warp,util

# build classification CNN for MNIST
def fullCNN(opt,image,dimShape,stddev):
	[conv1dim,conv2dim,conv3dim,conv4dim,fc5dim] = dimShape
	conv4fcDim = ((opt.H-4)//2-4)*((opt.W-4)//2-4)*conv4dim
	with tf.variable_scope("conv1"):
		weight,bias = createVariable([3,3,1,conv1dim],stddev)
		conv1 = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="VALID")+bias
		relu1 = tf.nn.relu(conv1)
	with tf.variable_scope("conv2"):
		weight,bias = createVariable([3,3,conv1dim,conv2dim],stddev)
		conv2 = tf.nn.conv2d(relu1,weight,strides=[1,1,1,1],padding="VALID")+bias
		relu2 = tf.nn.relu(conv2)
		maxpool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
	with tf.variable_scope("conv3"):
		weight,bias = createVariable([3,3,conv2dim,conv3dim],stddev)
		conv3 = tf.nn.conv2d(maxpool2,weight,strides=[1,1,1,1],padding="VALID")+bias
		relu3 = tf.nn.relu(conv3)
	with tf.variable_scope("conv4"):
		weight,bias = createVariable([3,3,conv3dim,conv4dim],stddev)
		conv4 = tf.nn.conv2d(relu3,weight,strides=[1,1,1,1],padding="VALID")+bias
		relu4 = tf.nn.relu(conv4)
	relu4vec = tf.reshape(relu4,[-1,conv4fcDim])
	with tf.variable_scope("fc5"):
		weight,bias = createVariable([conv4fcDim,fc5dim],stddev)
		fc5 = tf.matmul(relu4vec,weight)+bias
		relu5 = tf.nn.relu(fc5)
	with tf.variable_scope("fc6"):
		weight,bias = createVariable([fc5dim,10],stddev)
		fc6 = tf.matmul(relu5,weight)+bias
	return fc6

# build classification CNN for MNIST
def CNN(opt,image,dimShape,stddev):
	[conv1dim] = dimShape
	conv1fcDim = (opt.H-8)*(opt.W-8)*conv1dim
	with tf.variable_scope("conv1"):
		weight,bias = createVariable([9,9,1,conv1dim],stddev)
		conv1 = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="VALID")+bias
		relu1 = tf.nn.relu(conv1)
	relu1vec = tf.reshape(relu1,[-1,conv1fcDim])
	with tf.variable_scope("fc2"):
		weight,bias = createVariable([conv1fcDim,10],stddev)
		fc2 = tf.matmul(relu1vec,weight)+bias
	return fc2

# build Spatial Transformer (depth=4)
def STN(opt,ImWarp,p,STlayerN,dimShape,stddev):
	util.makeImageSummary("image",ImWarp,opt)
	for l in range(STlayerN):
		with tf.name_scope("ST{0}".format(l)):
			[STconv1dim,STconv2dim,STfc3dim] = dimShape
			STconv2fcDim = (opt.H-12)//2*(opt.W-12)//2*STconv2dim
			with tf.variable_scope("conv1"):
				weight,bias = createVariable([7,7,1,STconv1dim],stddev)
				STconv1 = tf.nn.conv2d(ImWarp,weight,strides=[1,1,1,1],padding="VALID")+bias
				STrelu1 = tf.nn.relu(STconv1)
			with tf.variable_scope("conv2"):
				weight,bias = createVariable([7,7,STconv1dim,STconv2dim],stddev)
				STconv2 = tf.nn.conv2d(STrelu1,weight,strides=[1,1,1,1],padding="VALID")+bias
				STrelu2 = tf.nn.relu(STconv2)
				STmaxpool2 = tf.nn.max_pool(STrelu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
			STmaxpool2vec = tf.reshape(STmaxpool2,[-1,STconv2fcDim])
			with tf.variable_scope("fc3"):
				weight,bias = createVariable([STconv2fcDim,STfc3dim],stddev)
				STfc3 = tf.matmul(STmaxpool2vec,weight)+bias
				STrelu3 = tf.nn.relu(STfc3)
			with tf.variable_scope("fc4"):
				weight,bias = createVariable([STfc3dim,opt.pDim],0,True)
				STfc4 = tf.matmul(STrelu3,weight)+bias
			warpMtrx = warp.vec2mtrxBatch(STfc4,opt)
			ImWarp = data.ImWarpIm(ImWarp,warpMtrx,opt)
			util.makeImageSummary("imageST{0}".format(l),ImWarp,opt)
			p = warp.compose(p,STfc4,opt)
	return ImWarp,p

# build compositional Spatial Transformer (depth=4)
def cSTN(opt,imageInput,p,STlayerN,dimShape,stddev):
	for l in range(STlayerN):
		with tf.name_scope("cST{0}".format(l)):
			warpMtrx = warp.vec2mtrxBatch(p,opt)
			ImWarp = data.imageWarpIm(imageInput,warpMtrx,opt)
			util.makeImageSummary("imageST{0}".format(l),ImWarp,opt)
			[STconv1dim,STconv2dim,STfc3dim] = dimShape
			STconv2fcDim = (opt.H-12)//2*(opt.W-12)//2*STconv2dim
			with tf.variable_scope("conv1"):
				weight,bias = createVariable([7,7,1,STconv1dim],stddev)
				STconv1 = tf.nn.conv2d(ImWarp,weight,strides=[1,1,1,1],padding="VALID")+bias
				STrelu1 = tf.nn.relu(STconv1)
			with tf.variable_scope("conv2"):
				weight,bias = createVariable([7,7,STconv1dim,STconv2dim],stddev)
				STconv2 = tf.nn.conv2d(STrelu1,weight,strides=[1,1,1,1],padding="VALID")+bias
				STrelu2 = tf.nn.relu(STconv2)
				STmaxpool2 = tf.nn.max_pool(STrelu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
			STmaxpool2vec = tf.reshape(STmaxpool2,[-1,STconv2fcDim])
			with tf.variable_scope("fc3"):
				weight,bias = createVariable([STconv2fcDim,STfc3dim],stddev)
				STfc3 = tf.matmul(STmaxpool2vec,weight)+bias
				STrelu3 = tf.nn.relu(STfc3)
			with tf.variable_scope("fc4"):
				weight,bias = createVariable([STfc3dim,opt.pDim],0,True)
				STfc4 = tf.matmul(STrelu3,weight)+bias
			p = warp.compose(p,STfc4,opt)
	warpMtrx = warp.vec2mtrxBatch(p,opt)
	ImWarp = data.imageWarpIm(imageInput,warpMtrx,opt)
	util.makeImageSummary("imageST{0}".format(STlayerN),ImWarp,opt)
	return ImWarp,p

# build compositional Spatial Transformer (recurrent,depth=4)
def ICSTN(opt,imageInput,p,STlayerN,dimShape,stddev):
	[STconv1dim,STconv2dim,STfc3dim] = dimShape
	STconv2fcDim = (opt.H-12)//2*(opt.W-12)//2*STconv2dim
	with tf.name_scope("cSTrecur"):
		with tf.variable_scope("conv1"):
			weight1,bias1 = createVariable([7,7,1,STconv1dim],stddev)
		with tf.variable_scope("conv2"):
			weight2,bias2 = createVariable([7,7,STconv1dim,STconv2dim],stddev)
		with tf.variable_scope("fc3"):
			weight3,bias3 = createVariable([STconv2fcDim,STfc3dim],stddev)
		with tf.variable_scope("fc4"):
			weight4,bias4 = createVariable([STfc3dim,opt.pDim],0,True)
	for l in range(STlayerN):
		with tf.name_scope("cSTrecur{0}".format(l)):
			warpMtrx = warp.vec2mtrxBatch(p,opt)
			ImWarp = data.imageWarpIm(imageInput,warpMtrx,opt)
			util.makeImageSummary("imageST{0}".format(l),ImWarp,opt)
			with tf.variable_scope("conv1"):
				STconv1 = tf.nn.conv2d(ImWarp,weight1,strides=[1,1,1,1],padding="VALID")+bias1
				STrelu1 = tf.nn.relu(STconv1)
			with tf.variable_scope("conv2"):
				STconv2 = tf.nn.conv2d(STrelu1,weight2,strides=[1,1,1,1],padding="VALID")+bias2
				STrelu2 = tf.nn.relu(STconv2)
				STmaxpool2 = tf.nn.max_pool(STrelu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
			STmaxpool2vec = tf.reshape(STmaxpool2,[-1,STconv2fcDim])
			with tf.variable_scope("fc3"):
				STfc3 = tf.matmul(STmaxpool2vec,weight3)+bias3
				STrelu3 = tf.nn.relu(STfc3)
			with tf.variable_scope("fc4"):
				STfc4 = tf.matmul(STrelu3,weight4)+bias4
			p = warp.compose(p,STfc4,opt)
	warpMtrx = warp.vec2mtrxBatch(p,opt)
	ImWarp = data.imageWarpIm(imageInput,warpMtrx,opt)
	util.makeImageSummary("imageST{0}".format(STlayerN),ImWarp,opt)
	return ImWarp,p

# auxiliary function for creating weight and bias
def createVariable(shape,stddev,zeroInit=False):
	if zeroInit:
		weight = tf.Variable(tf.zeros(shape),name="weight")
		bias = tf.Variable(tf.zeros([shape[-1]]),name="bias")
	else:
		weight = tf.Variable(tf.random_normal(shape,stddev=stddev),name="weight")
		bias = tf.Variable(tf.random_normal([shape[-1]],stddev=stddev),name="bias")
	return weight,bias
