import numpy as np
import scipy.linalg
import tensorflow as tf

# fit (affine) warp between two sets of points 
def fit(Xsrc,Xdst):
	ptsN = len(Xsrc)
	X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
						np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
	return pMtrx

# compute composition of warp parameters
def compose(opt,p,dp):
	with tf.name_scope("compose"):
		pMtrx = vec2mtrx(opt,p)
		dpMtrx = vec2mtrx(opt,dp)
		pMtrxNew = tf.matmul(dpMtrx,pMtrx)
		pMtrxNew /= pMtrxNew[:,2:3,2:3]
		pNew = mtrx2vec(opt,pMtrxNew)
	return pNew

# compute inverse of warp parameters
def inverse(opt,p):
	with tf.name_scope("inverse"):
		pMtrx = vec2mtrx(opt,p)
		pInvMtrx = tf.matrix_inverse(pMtrx)
		pInv = mtrx2vec(opt,pInvMtrx)
	return pInv

# convert warp parameters to matrix
def vec2mtrx(opt,p):
	with tf.name_scope("vec2mtrx"):
		O = tf.zeros([opt.batchSize])
		I = tf.ones([opt.batchSize])
		if opt.warpType=="translation":
			tx,ty = tf.unstack(p,axis=1)
			pMtrx = tf.transpose(tf.stack([[I,O,tx],[O,I,ty],[O,O,I]]),perm=[2,0,1])
		if opt.warpType=="similarity":
			pc,ps,tx,ty = tf.unstack(p,axis=1)
			pMtrx = tf.transpose(tf.stack([[I+pc,-ps,tx],[ps,I+pc,ty],[O,O,I]]),perm=[2,0,1])
		if opt.warpType=="affine":
			p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
			pMtrx = tf.transpose(tf.stack([[I+p1,p2,p3],[p4,I+p5,p6],[O,O,I]]),perm=[2,0,1])
		if opt.warpType=="homography":
			p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
			pMtrx = tf.transpose(tf.stack([[I+p1,p2,p3],[p4,I+p5,p6],[p7,p8,I]]),perm=[2,0,1])
	return pMtrx

# convert warp matrix to parameters
def mtrx2vec(opt,pMtrx):
	with tf.name_scope("mtrx2vec"):
		[row0,row1,row2] = tf.unstack(pMtrx,axis=1)
		[e00,e01,e02] = tf.unstack(row0,axis=1)
		[e10,e11,e12] = tf.unstack(row1,axis=1)
		[e20,e21,e22] = tf.unstack(row2,axis=1)
		if opt.warpType=="translation": p = tf.stack([e02,e12],axis=1)
		if opt.warpType=="similarity": p = tf.stack([e00-1,e10,e02,e12],axis=1)
		if opt.warpType=="affine": p = tf.stack([e00-1,e01,e02,e10,e11-1,e12],axis=1)
		if opt.warpType=="homography": p = tf.stack([e00-1,e01,e02,e10,e11-1,e12,e20,e21],axis=1)
	return p

# warp the image
def transformImage(opt,image,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(opt.refMtrx,axis=0),[opt.batchSize,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize,1,1]),[1,opt.H,opt.W])
		imageVec = tf.reshape(image,[-1,int(image.shape[-1])])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,int(image.shape[-1])])],axis=0)
		idxUL = (imageIdx*opt.H+YfloorInt)*opt.W+XfloorInt
		idxUR = (imageIdx*opt.H+YfloorInt)*opt.W+XceilInt
		idxBL = (imageIdx*opt.H+YceilInt)*opt.W+XfloorInt
		idxBR = (imageIdx*opt.H+YceilInt)*opt.W+XceilInt
		idxOutside = tf.fill([opt.batchSize,opt.H,opt.W],opt.batchSize*opt.H*opt.W)
		def insideImage(Xint,Yint):
			return (Xint>=0)&(Xint<opt.W)&(Yint>=0)&(Yint<opt.H)
		idxUL = tf.where(insideImage(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideImage(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideImage(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideImage(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[opt.batchSize,opt.H,opt.W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[opt.batchSize,opt.H,opt.W,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp

# warp the image
def transformCropImage(opt,imageFull,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(opt.bboxRefMtrx,axis=0),[opt.batchSize,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize,1,1]),[1,opt.H,opt.W])
		imageVec = tf.reshape(imageFull,[-1,int(imageFull.shape[-1])])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,int(imageFull.shape[-1])])],axis=0)
		idxUL = (imageIdx*opt.fullH+YfloorInt)*opt.fullW+XfloorInt
		idxUR = (imageIdx*opt.fullH+YfloorInt)*opt.fullW+XceilInt
		idxBL = (imageIdx*opt.fullH+YceilInt)*opt.fullW+XfloorInt
		idxBR = (imageIdx*opt.fullH+YceilInt)*opt.fullW+XceilInt
		idxOutside = tf.fill([opt.batchSize,opt.H,opt.W],opt.batchSize*opt.fullH*opt.fullW)
		def insideImage(Xint,Yint):
			return (Xint>=0)&(Xint<opt.fullW)&(Yint>=0)&(Yint<opt.fullH)
		idxUL = tf.where(insideImage(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideImage(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideImage(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideImage(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[opt.batchSize,opt.H,opt.W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[opt.batchSize,opt.H,opt.W,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp
