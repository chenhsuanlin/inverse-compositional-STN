import os
import tensorflow as tf

def mkdir(path):
	if not os.path.exists(path): os.mkdir(path)

# make image_summary from image batch
def makeImageSummary(tag,image,opt):
	with tf.name_scope("imageSummary"):
		blockSize = opt.visBlockSize
		imageSlice = tf.slice(image,[0,0,0,0],[blockSize**2,-1,-1,-1])
		imageOne = tf.batch_to_space(imageSlice,crops=[[0,0],[0,0]],block_size=blockSize)
		imagePermute = tf.reshape(imageOne,[opt.H,blockSize,opt.W,blockSize,1])
		imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
		imageBlocks = tf.reshape(imageTransp,[1,opt.H * blockSize,opt.W * blockSize,1])
		tf.summary.image(tag,imageBlocks)

# set optimizer for different learning rates
def setOptimizer(loss,learningRate,opt):
	varList = tf.global_variables()
	varListST = [v for v in varList if "ST" in v.name]
	varListOther = [v for v in varList if "ST" not in v.name]
	lrST,lrOther = tf.unstack(learningRate)
	gradients = tf.gradients(loss,varListST + varListOther)
	optimizerOther = tf.train.GradientDescentOptimizer(lrOther)
	gradientsOther = gradients[len(varListST):]
	trainStepOther = optimizerOther.apply_gradients(zip(gradientsOther,varListOther))
	if len(varListST) > 0:
		optimizerST = tf.train.GradientDescentOptimizer(lrST)
		gradientsST = gradients[:len(varListST)]
		trainStepST = optimizerST.apply_gradients(zip(gradientsST,varListST))
		trainStep = tf.group(trainStepST,trainStepOther)
	else:
		trainStep = trainStepOther
	return trainStep
