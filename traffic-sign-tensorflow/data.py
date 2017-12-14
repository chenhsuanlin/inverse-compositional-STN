import numpy as np
import scipy.linalg,scipy.misc
import os,time
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

import warp

# load GTSRB data
def loadGTSRB(opt,fname):
	if not os.path.exists(fname):
		# download and preprocess GTSRB dataset
		os.makedirs(os.path.dirname(fname))
		os.system("wget -O data/GTSRB_Final_Training_Images.zip http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip")
		os.system("wget -O data/GTSRB_Final_Test_Images.zip http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip")
		os.system("wget -O data/GTSRB_Final_Test_GT.zip http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip")
		os.system("cd data && unzip GTSRB_Final_Training_Images.zip")
		os.system("cd data && unzip GTSRB_Final_Test_Images.zip")
		os.system("cd data && unzip GTSRB_Final_Test_GT.zip")
		# training data
		print("preparing training data...")
		images,bboxes,labels = [],[],[]
		for c in range(43):
			prefix = "data/GTSRB/Final_Training/Images/{0:05d}".format(c)
			with open("{0}/GT-{1:05d}.csv".format(prefix,c)) as file:
				reader = csv.reader(file,delimiter=";")
				next(reader)
				for line in reader:
					img = plt.imread(prefix+"/"+line[0])
					rawH,rawW = img.shape[0],img.shape[1]
					scaleH,scaleW = float(opt.fullH)/rawH,float(opt.fullW)/rawW
					imgResize = scipy.misc.imresize(img,(opt.fullH,opt.fullW,3))
					images.append(imgResize)
					bboxes.append([float(line[3])*scaleW,float(line[4])*scaleH,
								   float(line[5])*scaleW,float(line[6])*scaleH])
					labels.append(int(line[7]))
		trainData = {
			"image": np.array(images),
			"bbox": np.array(bboxes),
			"label": np.array(labels)
		}
		# test data
		print("preparing test data...")
		images,bboxes,labels = [],[],[]
		prefix = "data/GTSRB/Final_Test/Images/"
		with open("data/GT-final_test.csv") as file:
			reader = csv.reader(file,delimiter=";")
			next(reader)
			for line in reader:
				img = plt.imread(prefix+"/"+line[0])
				rawH,rawW = img.shape[0],img.shape[1]
				scaleH,scaleW = float(opt.fullH)/rawH,float(opt.fullW)/rawW
				imgResize = scipy.misc.imresize(img,(opt.fullH,opt.fullW,3))
				images.append(imgResize)
				bboxes.append([float(line[3])*scaleW,float(line[4])*scaleH,
							   float(line[5])*scaleW,float(line[6])*scaleH])
				labels.append(int(line[7]))
		testData = {
			"image": np.array(images),
			"bbox": np.array(bboxes),
			"label": np.array(labels)
		}
		np.savez(fname,train=trainData,test=testData)
		os.system("rm -rf data/*.zip")
	GTSRB = np.load(fname)
	trainData = GTSRB["train"].item()
	testData = GTSRB["test"].item()
	return trainData,testData

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
			dXY = tf.expand_dims(tf.concat([dX,dY],1),-1)
			pPert = tf.matrix_solve_ls(J,dXY)[:,:,0]
	return pPert

# make training batch
def makeBatch(opt,data,PH):
	N = len(data["image"])
	randIdx = np.random.randint(N,size=[opt.batchSize])
	# put data in placeholders
	[image,label] = PH
	batch = {
		image: data["image"][randIdx]/255.0,
		label: data["label"][randIdx],
	}
	return batch

# evaluation on test set
def evalTest(opt,sess,data,PH,prediction,imagesEval=[]):
	N = len(data["image"])
	# put data in placeholders
	[image,label] = PH
	batchN = int(np.ceil(N/opt.batchSize))
	warped = [{},{}]
	count = 0
	for b in range(batchN):
		# use some dummy data (0) as batch filler if necessary
		if b!=batchN-1:
			realIdx = np.arange(opt.batchSize*b,opt.batchSize*(b+1))
		else:
			realIdx = np.arange(opt.batchSize*b,N)
		idx = np.zeros([opt.batchSize],dtype=int)
		idx[:len(realIdx)] = realIdx
		batch = {
			image: data["image"][idx]/255.0,
			label: data["label"][idx],
		}
		evalList = sess.run([prediction]+imagesEval,feed_dict=batch)
		pred = evalList[0]
		count += pred[:len(realIdx)].sum()
		if len(imagesEval)>0:
			imgs = evalList[1:]
			for i in range(len(realIdx)):
				if data["label"][idx[i]] not in warped[0]: warped[0][data["label"][idx[i]]] = []
				if data["label"][idx[i]] not in warped[1]: warped[1][data["label"][idx[i]]] = []
				warped[0][data["label"][idx[i]]].append(imgs[0][i])
				warped[1][data["label"][idx[i]]].append(imgs[1][i])
	accuracy = float(count)/N
	if len(imagesEval)>0:
		mean = [np.array([np.mean(warped[0][l],axis=0) for l in warped[0]]),
				np.array([np.mean(warped[1][l],axis=0) for l in warped[1]])]
		var = [np.array([np.var(warped[0][l],axis=0) for l in warped[0]]),
			   np.array([np.var(warped[1][l],axis=0) for l in warped[1]])]
	else: mean,var = None,None
	return accuracy,mean,var
