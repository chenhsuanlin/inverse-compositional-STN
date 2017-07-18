import numpy as np
import argparse
import warp

def set():

	# parse input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("type",metavar="TYPE",help="type of network",choices=["CNN","STN","cSTN","ICSTN"])
	parser.add_argument("--group",default="0",help="name for group")
	parser.add_argument("--model",default="test",help="name for model instance")
	parser.add_argument("--recurN",type=int,default=4,help="number of recurrent transformations (for IC-STN)")
	parser.add_argument("--lr",type=float,default=1e-2,help="learning rate for classification network")
	parser.add_argument("--lrST",type=float,default=1e-4,help="learning rate for geometric predictor")
	parser.add_argument("--batchSize",type=int,default=200,help="batch size for SGD")
	parser.add_argument("--maxIter",type=int,default=100000,help="maximum number of training iterations")
	parser.add_argument("--warpType",metavar="WARPTYPE",help="type of warp function on images",default="homography",
									 choices=["translation","similarity","affine","homography"])
	parser.add_argument("--resume",type=int,default=0,help="resume from iteration number")
	parser.add_argument("--gpu",type=int,default=0,help="ID of GPU device (if there are multiple)")
	opt = parser.parse_args()

	opt.warpScale = {"pert":0.25,"trans":0.25}
	opt.baseLR,opt.baseLRST = opt.lr,opt.lrST
	# --- below are automatically set ---
	opt.H,opt.W = 28,28
	opt.visBlockSize = 12
	opt.GPUdevice = "/gpu:{0}".format(opt.gpu)
	opt.pDim = 2 if opt.warpType == "translation" else \
				4 if opt.warpType == "similarity" else \
				6 if opt.warpType == "affine" else \
				8 if opt.warpType == "homography" else None
	opt.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
	opt.Im4pts = np.array([[0,0],[0,opt.H-1],[opt.W-1,opt.H-1],[opt.W-1,0]],dtype=np.float32)
	opt.warpGTmtrx = warp.fit(opt.canon4pts,opt.Im4pts,"affine")
	if opt.type=="CNN": opt.baseLRST = 0

	return opt
