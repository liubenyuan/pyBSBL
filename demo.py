# This demo shows how to set input parameters of BSBL-BO and BSBL-FM
# for noisy experiments when block partition is known.
#
# modified by liubenyuan for the purpose of illustrating the BSBL-BO.py
#     liubenyuan@gmail.com
# 
import numpy as np
import scipy.linalg as lp
import matplotlib.pyplot as plt
import bsbl

# problem dimension
M = 128          # row number of the dictionary matrix 
N = 256          # column number

blkNum = 6       # nonzero block number
blkLen = 16      # block length

SNR = 15         # Signal-to-noise ratio
iterNum = 1      # number of experiments (100)
r = 0.95         # intra-correlation

#
np.random.seed(1985)

# Generate the known matrix with columns draw uniformly from the surface of a unit hypersphere
Phi = np.random.randn(M,N)
for i in range(Phi.shape[1]):
    Phi[:,i] = Phi[:,i] / np.sqrt(np.sum(Phi[:,i]**2))

# generate nonzero block coefficients
blks = np.zeros((blkNum,blkLen))
blks[:,0] = np.random.randn(blkNum)
for i in range(1,blkLen):
    blks[:,i] = r*blks[:,i-1] + \
                np.sqrt(1.0 - r**2) * np.random.randn(blkNum)

#===========================================================================
# put blocks at random locations and align with block partition (no overlap) 
#===========================================================================
blk_start_loc = np.arange(0,N,blkLen)
nblock = blk_start_loc.shape[0]
ind_block = np.random.permutation(nblock)
block_slice = [blk_start_loc[i] + np.arange(blkLen) for i in xrange(nblock)]
#
x = np.zeros(N, dtype='float')
for i in range(blkNum):
    x[block_slice[ind_block[i]]] = blks[i,:]

# noiseless signal
y_clean = np.dot(Phi, x)

# add noise
stdnoise = y_clean.std() * (10**(-SNR/20.))
noise = np.random.rand(M) * stdnoise
y = y_clean + noise

#======================================================================
#            Algorithm Comparison
#======================================================================
ind = (np.abs(x)>0).nonzero()[0]

#  Benchmark
supt = ind
print (Phi[:,supt].shape)
x_ls = np.dot(lp.pinv(Phi[:,supt]), y)
x0 = np.zeros(N)
x0[supt] = x_ls
#
mse_bench = (lp.norm(x - x0)/lp.norm(x))**2

# BSBL-BO
clf = bsbl.bo(learn_lambda=1, learn_type=1, lambda_init=1e-3, 
              epsilon=1e-5, max_iters=200)
x1 = clf.fit_transform(Phi, y, blk_start_loc)
#
mse_bo = (lp.norm(x - x1)/lp.norm(x))**2

# BSBL-FM


# visualize
plt.figure()
plt.plot(x, linewidth=3)
plt.plot(x0, 'r-', linewidth=2)
plt.plot(x1, 'g-')
plt.legend({'MSE (LS) = '+str(mse_bench), 'MSE (BO) = '+str(mse_bo)},
            loc='lower left')
plt.axis('tight')


