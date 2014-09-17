# Example showing the ability of BSBL to recover non-sparse signals.
#
# The signal to recover is a real-world fetal ECG data, which consists two
# peaks of fetal ECG and one peak of maternal ECG.
#
# The goal is to recover the signal without distorting the two peaks of fetal
# ECG (the two peaks locate around at 65 and 180).
# 
# Details can be found in the paper:
#    Zhilin Zhang, Tzyy-Ping Jung, Scott Makeig, Bhaskar D. Rao, 
#    Low Energy Wireless Body-Area Networks for Fetal ECG Telemonitoring 
#    via the Framework of Block Sparse Bayesian Learning, submitted to 
#    IEEE Trans. on Biomedical Engineering, 2012. [Online] 
# http://arxiv.org/pdf/1205.1287v1
#
# Author: liubenyuan (liubenyuan@gmail.com)
#

import sys
import numpy as np
import scipy.linalg as lp
import scipy.io as io
import scipy.fftpack as sf
import matplotlib.pyplot as plt
#
sys.path.append('../')
import bsbl
#import bsbl.fm

N = 250;
M = 125; 

# load fetal ECG data
data = io.loadmat('ECGsegment.mat');
x = data['ecg'].ravel()
print ('ECG signal x shape=(%d)' % x.shape)

# load a sparse binary matrix. The matrix was randomly generated, each
# column of which has only 15 entries of 1. Note that other similar sparse 
# binary matrix works well also.
data = io.loadmat('Phi.mat');
Phi = data['Phi']
print ('Sparse Sensing matrix Phi, shape=(%d,%d)' % Phi.shape)

#=========================== Compress the ECG data ====================
y = np.dot(Phi, x)
print ('y shape = (%d)' % y.shape)

#========================== Method 1 ======================================
#  Reconstruct the ECG data by BSBL-BO directly  
#==========================================================================

# Define block partition
blkLen = 25
groupStartLoc = np.arange(0,N,blkLen);

# Run BSBL-BO which exploits the intra-block correlation
# Note: The ECG data is non-sparse. To reconstruct non-sparse signals, you
# need to set the input argument 'prune_gamma' to any non-positive values
# (0, -1, -10, etc), namely turning off this pruning mechanism in SBL. But
# when you reconstruct sparse signals, you can use the default value (1e-2).
# 
clf = bsbl.bo(verbose=1, learn_type=1, learn_lambda=2,
              prune_gamma=-1, epsilon=1e-8, max_iters=20)
x1 = clf.fit_transform(Phi, y, blk_start_loc=groupStartLoc)
#
mse_bo = 10*np.log10((lp.norm(x - x1)**2)/lp.norm(x)**2)
print ('BSBL-BO exit on %d loop' % clf.count)

plt.figure()
plt.plot(x,linewidth=3)
plt.plot(x1,'r-')
plt.title('MSE of BO (directly) is '+ str(mse_bo) + 'dB')
plt.legend({'Original', 'Recovered'})

#=========================== Second Method ==============================
# First recover the signal's coefficients in the DCT domain;
# Then recover the signal using the DCT ceofficients and the DCT basis
#=========================================================================
A = np.zeros([M,N],dtype='float')
for k in xrange(M):
    dct_k = sf.dct(Phi[k,:].astype('float'),norm='ortho')
    A[k,:] = dct_k.copy();
#
clf = bsbl.bo(verbose=1, learn_type=1, learn_lambda=2,
              prune_gamma=-1, epsilon=1e-8, max_iters=16)
rev_dct_coeff = clf.fit_transform(A, y, blk_start_loc=groupStartLoc)
# IDCT only accept 'row' vector !
x2 = sf.idct(rev_dct_coeff,norm='ortho');
#
mse_bo_dct = 10*np.log10((lp.norm(x-x2)**2)/lp.norm(x)**2)
print ('BSBL-BO exit on %d loop' % clf.count)

plt.figure()
plt.plot(x,linewidth=3)
plt.plot(x2,'r-')
plt.title('MSE of BO (directly) is '+ str(mse_bo_dct) + 'dB')
plt.legend({'Original', 'Recovered'})

plt.figure()
plt.plot(sf.dct(x, norm='ortho'), linewidth=3)
plt.plot(rev_dct_coeff, 'r-')
plt.title('wtrue and w in DCT domain')
