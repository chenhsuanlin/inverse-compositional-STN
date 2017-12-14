import numpy as np
import time,os,sys
import argparse
import util

print(util.toYellow("======================================================="))
print(util.toYellow("train.py (training on MNIST)"))
print(util.toYellow("======================================================="))

import torch
import data,graph,warp,util
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=True)

# create directories for model output
util.mkdir("models_{0}".format(opt.group))

print(util.toMagenta("building network..."))
with torch.cuda.device(0):
	classifier = graph.FullCNN(opt)
	# ------ define loss ------
	loss = torch.nn.CrossEntropyLoss()
	# ------ optimizer ------
	optimList = [{ "params": classifier.parameters(), "lr": opt.lrC }]
	optim = torch.optim.SGD(optimList)

# load data
print(util.toMagenta("loading MNIST dataset..."))
trainData,validData,testData = data.loadMNIST("data/MNIST.npz")

print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
with torch.cuda.device(0):
	classifier.train()
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		# make training batch
		batch = data.makeBatch(opt,trainData)
		image = batch["image"].unsqueeze(dim=1)
		label = batch["label"]
		# generate perturbation
		pInit = util.toTorch(np.zeros([opt.batchSize,opt.warpDim],dtype=np.float32))
		pInitMtrx = warp.vec2mtrx(opt,pInit)
		# forward/backprop through network
		optim.zero_grad()
		imagePert = warp.transformImage(opt,image,pInitMtrx)
			# print((imagePert-image).data.cpu().mean(),(imagePert-image).data.cpu().var())
			# img = image.data.cpu().numpy()
			# imgpert = imagePert.data.cpu().numpy()
			# util.imsave("temp1.png",img[0].squeeze())
			# util.imsave("temp2.png",imgpert[0].squeeze())
			# for i in range(100): util.imsave("temp/{0}_1.png".format(i),1-img[i].squeeze())
			# for i in range(100): util.imsave("temp/{0}_2.png".format(i),1-imgpert[i].squeeze())
			# print(imagePert[0,0],image[0,0])
			# assert(False)
		# forward/backprop through network
		optim.zero_grad()
		output = classifier(opt,imagePert)
		train_loss = loss(output,label)
		train_loss.backward()
		# run one step
		optim.step()
		if (i+1)%100==0:
			print("it. {0}/{1} loss={3}, time={2}"
				.format(util.toCyan("{0}".format(i+1)),
						opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toRed("{0:.4f}".format(train_loss.data[0]))))
	assert(False)
