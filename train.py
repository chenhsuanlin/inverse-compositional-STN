import numpy as np
import time,os,sys
import argparse

import options
print("setting configurations...")
opt = options.set()

import tensorflow as tf
import data,graph,graphST,warp

print("=======================================================")
print("train.py (training on MNIST)")
print("=======================================================")

# load data
print("loading MNIST dataset...")
trainData,validData,testData = data.loadMNIST("data/MNIST.npz")

# create directories for model output
util.mkdir("models_{0}".format(opt.group))
util.mkdir("models_{0}/interm".format(opt.group))
util.mkdir("models_{0}/final".format(opt.group))

print("training model {0}...".format(opt.model))
print("------------------------------------------")
print("warpScale: (pert) {0} (trans) {1}".format(opt.warpScale["pert"],opt.warpScale["trans"]))
print("warpType: {0}".format(opt.warpType))
print("batchSize: {0}".format(opt.batchSize))
print("GPU device: {0}".format(opt.gpu))
print("------------------------------------------")

tf.reset_default_graph()
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
# build graph
with tf.device(opt.GPUdevice):
	# generate training data on the fly
	imageRawBatch = tf.placeholder(tf.float32,shape=[None,28,28],name="image")
	pInitBatch = data.genPerturbations(opt)
	pInitMtrxBatch = warp.vec2mtrxBatch(pInitBatch,opt)
	ImBatch = data.imageWarpIm(imageRawBatch,pInitMtrxBatch,opt,name=None)
	# build network
	if opt.type=="CNN":
		outputBatch = graph.fullCNN(opt,ImBatch,[3,6,9,12,48],0.1)
	elif opt.type=="STN":
		ImWarpBatch,pBatch = graph.STN(opt,ImBatch,pInitBatch,1,[4,8,48],0.01)
		outputBatch = graph.CNN(opt,ImWarpBatch,[3],0.03)
	elif opt.type=="cSTN":
		ImWarpBatch,pBatch = graph.cSTN(opt,imageRawBatch,pInitBatch,1,[4,8,48],0.01)
		outputBatch = graph.CNN(opt,ImWarpBatch,[3],0.03)
	elif opt.type=="ICSTN":
		ImWarpBatch,pBatch = graph.ICSTN(opt,imageRawBatch,pInitBatch,opt.recurN,[4,8,48],0.01)
		outputBatch = graph.CNN(opt,ImWarpBatch,[3],0.03)
	# define loss/optimizer/summaries
	imageSummaries = tf.summary.merge_all()
	labelBatch = tf.placeholder(tf.float32,shape=[None,10],name="label")
	softmaxLoss = tf.nn.softmax_cross_entropy_with_logits(logits=outputBatch,labels=labelBatch)
	loss = tf.reduce_mean(softmaxLoss)
	lossSummary = tf.summary.scalar("training loss",loss)
	learningRate = tf.placeholder(tf.float32,shape=[2])
	trainStep = util.setOptimizer(loss,learningRate,opt)
	softmax = tf.nn.softmax(outputBatch)
	prediction = tf.equal(tf.argmax(softmax,1),tf.argmax(labelBatch,1))

print("starting backpropagation...")
trainN = len(trainData["image"])
timeStart = time.time()
# define some more summaries
tfSaver,tfSaverInterm,tfSaverFinal = tf.train.Saver(max_to_keep=10), \
									 tf.train.Saver(max_to_keep=10), \
									 tf.train.Saver()
testErrorPH = tf.placeholder(tf.float32,shape=[])
testErrorSummary = tf.summary.scalar("test error",testErrorPH)
tfSummaryWriter = tf.summary.FileWriter("summary_{1}/{0}".format(opt.model,opt.group))
resumeIterN = opt.resume
maxIterN = opt.maxIter
with tf.Session(config=tfConfig) as sess:
	if resumeIterN==0:
		sess.run(tf.global_variables_initializer())
	else:
		tfSaver.restore(sess,"models_{2}/{0}_it{1}k.ckpt".format(opt.model,resumeIterN//1000,opt.group))
		print("resuming from iteration {0}...".format(resumeIterN))
	tfSummaryWriter.add_graph(sess.graph)
	# training loop
	for i in range(resumeIterN,maxIterN):
		currLearningRate = opt.baseLRST,opt.baseLR  # this can be modified to be scheduled learning rates
		randIdx = np.random.randint(trainN,size=[opt.batchSize])
		trainBatch = {
			imageRawBatch: trainData["image"][randIdx],
			labelBatch: trainData["label"][randIdx],
			learningRate: currLearningRate
		}
		# run one step
		_,trainBatchLoss,summary = sess.run([trainStep,loss,lossSummary],feed_dict=trainBatch)
		if (i+1)%20==0:
			tfSummaryWriter.add_summary(summary,i+1)
		if (i+1)%100==0:
			print("it. {0}/{1} (lr={5:.2e},{4:.2e}),loss={2:.4f},time={3:.4f}"
				  .format(i+1,maxIterN,trainBatchLoss,time.time()-timeStart,currLearningRate[0],currLearningRate[1]))
		if (i+1)%2000==0:
			# update image summaries
			if imageSummaries is not None:
				summary = sess.run(imageSummaries,feed_dict=trainBatch)
				tfSummaryWriter.add_summary(summary,i+1)
			# evaluate on validation and test sets
			testAccuracy = data.evaluate(testData,imageRawBatch,pInitBatch,labelBatch,prediction,sess,opt)
			testError = (1-testAccuracy)*100
			summary = sess.run(testErrorSummary,feed_dict={testErrorPH: testError})
			tfSummaryWriter.add_summary(summary,i+1)
			# save model
			savePath = tfSaver.save(sess,"models_{2}/{0}_it{1}k.ckpt".format(opt.model,(i+1)//1000,opt.group))
			print("model saved: {0}".format(savePath))
		if (i+1)%10000==0:
			# save intermediate model
			tfSaverInterm.save(sess,"models_{2}/interm/{0}_it{1}k.ckpt".format(opt.model,(i+1)//1000,opt.group))
	# save final model
	tfSaverFinal.save(sess,"models_{1}/final/{0}.ckpt".format(opt.model,opt.group))
print("======= backpropagation done =======")
