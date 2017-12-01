import numpy as np
import time,os,sys
import argparse
import util

print(util.toYellow("======================================================="))
print(util.toYellow("train.py (training on MNIST)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data,graph,warp,util
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=True)

# create directories for model output
util.mkdir("models_{0}".format(opt.group))

print(util.toMagenta("building graph..."))
tf.reset_default_graph()
# build graph
with tf.device("/gpu:0"):
	# ------ define input data ------
	imageFull = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.fullH,opt.fullW,3])
	imageMean,imageVar = tf.nn.moments(imageFull,axes=[1,2],keep_dims=True)
	imageFullNormalize = (imageFull-imageMean)/tf.sqrt(imageVar)
	label = tf.placeholder(tf.int64,shape=[opt.batchSize])
	PH = [imageFull,label]
	# ------ generate perturbation ------
	pInit = data.genPerturbations(opt)
	pInitMtrx = warp.vec2mtrx(opt,pInit)
	# ------ build network ------
	imagePert = warp.transformCropImage(opt,imageFullNormalize,pInitMtrx)
	imagePertRescale = imagePert*tf.sqrt(imageVar)+imageMean
	if opt.netType=="CNN":
		output = graph.fullCNN(opt,imagePert)
	elif opt.netType=="STN":
		imageWarpAll = graph.STN(opt,imagePert)
		imageWarp = imageWarpAll[-1]
		output = graph.CNN(opt,imageWarp)
		imageWarpRescale = imageWarp*tf.sqrt(imageVar)+imageMean
	elif opt.netType=="IC-STN":
		imageWarpAll = graph.ICSTN(opt,imageFullNormalize,pInit)
		imageWarp = imageWarpAll[-1]
		output = graph.CNN(opt,imageWarp)
		imageWarpRescale = imageWarp*tf.sqrt(imageVar)+imageMean
	softmax = tf.nn.softmax(output)
	labelOnehot = tf.one_hot(label,opt.labelN)
	prediction = tf.equal(tf.argmax(softmax,1),label)
	# ------ define loss ------
	softmaxLoss = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labelOnehot)
	loss = tf.reduce_mean(softmaxLoss)
	# ------ optimizer ------
	lrGP_PH,lrC_PH = tf.placeholder(tf.float32,shape=[]),tf.placeholder(tf.float32,shape=[])
	optim = util.setOptimizer(opt,loss,lrGP_PH,lrC_PH)
	# ------ generate summaries ------
	summaryImageTrain = []
	summaryImageTest = []
	if opt.netType=="STN" or opt.netType=="IC-STN":
		for l in range(opt.warpN+1):
			summaryImageTrain.append(util.imageSummary(opt,imageWarpAll[l]*tf.sqrt(imageVar)+imageMean,"TRAIN_warp{0}".format(l),opt.H,opt.W))
			summaryImageTest.append(util.imageSummary(opt,imageWarpAll[l]*tf.sqrt(imageVar)+imageMean,"TEST_warp{0}".format(l),opt.H,opt.W))
		summaryImageTrain = tf.summary.merge(summaryImageTrain)
		summaryImageTest = tf.summary.merge(summaryImageTest)
	summaryLossTrain = tf.summary.scalar("TRAIN_loss",loss)
	testErrorPH = tf.placeholder(tf.float32,shape=[])
	testImagePH = tf.placeholder(tf.float32,shape=[opt.labelN,opt.H,opt.W,3])
	summaryErrorTest = tf.summary.scalar("TEST_error",testErrorPH)
	if opt.netType=="STN" or opt.netType=="IC-STN":
		summaryMeanTest0 = util.imageSummaryMeanVar(opt,testImagePH,"TEST_mean_init",opt.H,opt.W)
		summaryMeanTest1 = util.imageSummaryMeanVar(opt,testImagePH,"TEST_mean_warped",opt.H,opt.W)
		summaryVarTest0 = util.imageSummaryMeanVar(opt,testImagePH,"TEST_var_init",opt.H,opt.W)
		summaryVarTest1 = util.imageSummaryMeanVar(opt,testImagePH,"TEST_var_warped",opt.H,opt.W)

# load data
print(util.toMagenta("loading GTSRB dataset..."))
trainData,testData = data.loadGTSRB(opt,"data/GTSRB.npz")

# prepare model saver/summary writer
saver = tf.train.Saver(max_to_keep=20)
summaryWriter = tf.summary.FileWriter("summary_{0}/{1}".format(opt.group,opt.model))

print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	summaryWriter.add_graph(sess.graph)
	if opt.fromIt!=0:
		util.restoreModel(opt,sess,saver,opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		lrGP = opt.lrGP*opt.lrGPdecay**(i//opt.lrGPstep)
		lrC = opt.lrC*opt.lrCdecay**(i//opt.lrCstep)
		# make training batch
		batch = data.makeBatch(opt,trainData,PH)
		batch[lrGP_PH] = lrGP
		batch[lrC_PH] = lrC
		# run one step
		_,l = sess.run([optim,loss],feed_dict=batch)
		if (i+1)%100==0:
			print("it. {0}/{1}  lr={3}(GP),{4}(C), loss={5}, time={2}"
				.format(util.toCyan("{0}".format(i+1)),
						opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("{0:.0e}".format(lrGP)),
						util.toYellow("{0:.0e}".format(lrC)),
						util.toRed("{0:.4f}".format(l))))
		if (i+1)%100==0:
			summaryWriter.add_summary(sess.run(summaryLossTrain,feed_dict=batch),i+1)
		if (i+1)%500==0 and (opt.netType=="STN" or opt.netType=="IC-STN"):
			summaryWriter.add_summary(sess.run(summaryImageTrain,feed_dict=batch),i+1)
			summaryWriter.add_summary(sess.run(summaryImageTest,feed_dict=batch),i+1)
		if (i+1)%1000==0:
			# evaluate on test set
			if opt.netType=="STN" or opt.netType=="IC-STN":
				testAcc,testMean,testVar = data.evalTest(opt,sess,testData,PH,prediction,imagesEval=[imagePert,imageWarp])
			else:
				testAcc,_,_ = data.evalTest(opt,sess,testData,PH,prediction)
			testError = (1-testAcc)*100
			summaryWriter.add_summary(sess.run(summaryErrorTest,feed_dict={testErrorPH:testError}),i+1)
			if opt.netType=="STN" or opt.netType=="IC-STN":
				summaryWriter.add_summary(sess.run(summaryMeanTest0,feed_dict={testImagePH:testMean[0]}),i+1)
				summaryWriter.add_summary(sess.run(summaryMeanTest1,feed_dict={testImagePH:testMean[1]}),i+1)
				summaryWriter.add_summary(sess.run(summaryVarTest0,feed_dict={testImagePH:testVar[0]}),i+1)
				summaryWriter.add_summary(sess.run(summaryVarTest1,feed_dict={testImagePH:testVar[1]}),i+1)
		if (i+1)%10000==0:
			util.saveModel(opt,sess,saver,i+1)
			print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.model,i+1)))

print(util.toYellow("======= TRAINING DONE ======="))
