# regression y=Xw using block sparse Bayesian learning framework
#
# {y,X} are known, and w is assumed to be 'sparse' or 'block sparse'
# the indices of the non-zero blocks can be either known or unknown
# 
# Authors: Benyuan Liu <liubenyuan@gmail.com>
# License: BSD 3 Clause
#
# For the BSBL-BO algorithm: 
# 
#@article{zhang2013extension, 
#    author={Zhang, Z. and Rao, B.D.}, 
#    journal={Signal Processing, IEEE Transactions on}, 
#    title={Extension of SBL Algorithms for the Recovery of Block Sparse Signals With Intra-Block Correlation}, 
#    year={2013}, 
#    month={April}, 
#    volume={61}, 
#    number={8}, 
#    pages={2009-2015}, 
#    doi={10.1109/TSP.2013.2241055}, 
#    ISSN={1053-587X},} 
#
# For the BSBL-FM algorithm:
# 
#@article{liu2014energy,
#    author = "Benyuan Liu and Zhilin Zhang and Gary Xu and Hongqi Fan and Qiang Fu",
#    title = "Energy efficient telemonitoring of physiological signals via compressed sensing: A fast algorithm and power consumption evaluation ",
#    journal = "Biomedical Signal Processing and Control ",
#    volume = "11",
#    number = "0",
#    pages = "80 - 88",
#    year = "2014",
#    issn = "1746-8094",
#    doi = "http://dx.doi.org/10.1016/j.bspc.2014.02.010",
#    url = "http://www.sciencedirect.com/science/article/pii/S1746809414000366",
#    }
#
# For the application of wireless telemonitoring via CS:
#
#@article{zhang2013compressed, 
#    author={Zhilin Zhang and Tzyy-Ping Jung and Makeig, S. and Rao, B.D.}, 
#    journal={Biomedical Engineering, IEEE Transactions on}, 
#    title={Compressed Sensing for Energy-Efficient Wireless Telemonitoring of Noninvasive Fetal ECG Via Block Sparse Bayesian Learning}, 
#    year={2013}, 
#    month={Feb}, 
#    volume={60}, 
#    number={2}, 
#    pages={300-309}, 
#    doi={10.1109/TBME.2012.2226175}, 
#    ISSN={0018-9294},}
#
from __future__ import print_function

import numpy as np
import scipy.linalg as lp

# vector to column 2D vector
def v2m(v):
    return v.reshape((v.shape[0],1))

# ravel list of arrays
def ravel_list(d):
    r = np.array([], dtype='int')
    for i in xrange(d.shape[0]):
        r = np.r_[r,d[i]]
    return r

# extract block spacings
def block_parse(blk_start_loc, N):
    blk_len_list = np.r_[blk_start_loc[1:], N] - blk_start_loc
    is_equal_block = (np.sum(np.abs(blk_len_list - blk_len_list.mean())) == 0)
    return blk_len_list, is_equal_block

# exploit AR(1) correlation in Covariance matrices
def coeff_r(Cov, gamma, index):
    r0 = 0.
    r1 = 0.
    for i in index:
        temp = Cov[i]/gamma[i]
        r0 += temp.trace().mean()
        r1 += temp.trace(offset=1).mean()
    # this method tend to under estimate the correlation
    r = 1.1 * r1/r0
    if (np.abs(r) >= 0.99):
        r = 0.99*np.sign(r)
    return r

class bo:
    """
    # BSBL-BO : Bound Optimization Algos of BSBL framework
    #
    # Recover block sparse signal (1D) exploiting intra-block correlation, 
    # given the block partition.
    #
    #   The algorithm solves the inverse problem for the block sparse
    #               model with known block partition:
    #                        y = X * w + v
    #
    #   X : array, shape = (n_samples, n_features)
    #   Training vectors.
    #
    #   y : array, shape = (n_samples)
    #   Target values for training vectors
    #
    #   w : array, shape = (n_features)
    #   sparse/block sparse weight vector
    #
    #   'learn_type'   : learn_type = 0: Ignore intra-block correlation
    #                    learn_type = 1: Exploit intra-block correlation 
    #                    [ Default: learn_type = 1 ]
    #   'verbose'      : debuging information.
    #   'epsilon'      : convergence criterion
    #   'prune_gamma'  : threshold to prune out small gamma_i 
    #                    (generally, 10^{-3} or 10^{-2})
    #   'max_iters'    : Maximum number of iterations.
    #                    [ Default value: max_iters = 500 ]
    #   'learn_lambda' : (1) if (SNR<10dB), learn_lambda=1
    #                    (2) if (SNR>10dB), learn_lambda=2
    #                    (3) if noiseless, learn_lambda=0
    #                    [ Default value: learn_lambda=2]
    #
    """

    # constructor
    def __init__(self, learn_lambda=2,
                  epsilon=1e-8, max_iters=500, verbose=0,
                  learn_type=1, prune_gamma=1e-2):
        self.learn_lambda = learn_lambda
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.verbose = verbose
        self.learn_type = learn_type
        self.prune_gamma = prune_gamma
        #
        if verbose:
            self.print_vars()
    
    # fit y                 
    def fit_transform(self, X, y, blk_start_loc=None):
        #
        self.lamb = 1e-2 * y.std()
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc==None:
            blkLen = int(N/16.)
            blk_start_loc = np.arange(0,N,blkLen)
        blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        # init variables
        nblock      = blk_start_loc.shape[0]
        w        = np.zeros(N,dtype='float')
        Sigma0      = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Sigma_w     = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Cov_x       = [np.identity(blk_len_list[i]) for i in range(nblock)]
        B           = [np.identity(blk_len_list[i]) for i in range(nblock)]
        invB        = [np.identity(blk_len_list[i]) for i in range(nblock)]
        block_slice = [blk_start_loc[i] + np.arange(blk_len_list[i]) for i in xrange(nblock)]
        gamma       = np.ones(nblock, dtype='float')
        HX          = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Hy          = [np.zeros(blk_len_list[i]) for i in range(nblock)]
        # loops
        for count in xrange(self.max_iters):
            # prune weights as their hyperparameter goes to zero
            # index -- 0:unused, 1:used
            index = np.argwhere(gamma > self.prune_gamma).ravel()
            if (index.shape[0] == 0):
                self.print_zero_vector()
                raise TypeError('w is a zero-vector, exiting.')
            # calculate XBX^T
            XBX = np.zeros((M,M), dtype=float)
            for i in index:
                Xi = X[:, block_slice[i]]
                XBX += np.dot(np.dot(Xi, Sigma0[i]), Xi.T)
            invXBX = lp.inv(XBX + self.lamb * np.identity(M))
            #
            for i in index:
                Xi = X[:, block_slice[i]]
                Hi = np.dot(Xi.T, invXBX)
                Hy[i] = np.dot(Hi, y)
                HX[i] = np.dot(Hi, Xi)
            # now we update basis
            w_old = w.copy()
            for i in index:
                seg = block_slice[i]
                w[seg] = np.dot(Sigma0[i], Hy[i])
                Sigma_w[i] = Sigma0[i] - np.dot(np.dot(Sigma0[i], HX[i]), Sigma0[i])
                mu_v = v2m(w[seg])
                Cov_x[i] = Sigma_w[i] + np.dot(mu_v, mu_v.T)
            
            #=========== Learn correlation structure in blocks ===========
            # 0: do not consider correlation structure in each block
            # 1: constrain all the blocks have the same correlation structure
            if self.learn_type == 1:
                r = coeff_r(Cov_x, gamma, index)
                if self.is_equal_block:
                    jup = np.arange(Cov_x[0].shape[0])
                    bs = r**jup
                    B0 = lp.toeplitz(bs)
                    invB0 = lp.inv(B0)
                    for i in index:
                        B[i] = B0
                        invB[i] = invB0
                else:
                    for i in index:
                        jup = np.arange(B[i].shape[0])
                        bs = r**jup
                        B[i] = lp.toeplitz(bs)
                        invB[i] = lp.inv(B[i])
            
            # estimate gammas
            gamma_old = gamma.copy()
            for i in index:
                denom = np.sqrt(np.dot(HX[i], B[i]).trace())
                gamma[i] = gamma_old[i] * lp.norm(np.dot(lp.sqrtm(B[i]), Hy[i])) / denom
                Sigma0[i] = B[i] * gamma[i]
            # estimate lambda
            if self.learn_lambda == 1:
                lambComp = 0.
                for i in index:
                    Xi = X[:,block_slice[i]];
                    lambComp += np.dot(np.dot(Xi, Sigma_w[i]), Xi.T).trace()
                self.lamb = lp.norm(y - np.dot(X, w))**2./N + lambComp/N; 
            elif self.learn_lambda == 2:
                lambComp = 0.
                for i in index:
                    lambComp += np.dot(Sigma_w[i], invB[i]).trace() / gamma_old[i]
                self.lamb = lp.norm(y - np.dot(X, w))**2./N + self.lamb * (w.size - lambComp)/N                    
                    
            #================= Check stopping conditions, eyc. ==============
            dmu = (np.abs(w_old - w)).max(0); # only SMV currently
            if (dmu < self.epsilon):
                break;
            if (count >= self.max_iters):
                break;
        # exit
        self.count = count + 1
        # let's convert the backyard:
        return w
        
    # print parameters
    def print_vars(self):
        print ('----------------------------INFO------------------------------')
        print ('apply lambda learning rule (learn_lambda) = %d' % self.learn_lambda)
        print ('BSBL algorithm exit criterion   (epsilon) = %g' % self.epsilon)
        print ('BSBL maximum iterations       (max_iters) = %d' % self.max_iters)
        print ('intra-block correlation      (learn_type) = %d' % self.learn_type)
        print ('Gamma pruning rules         (prune_gamma) = %g' % self.prune_gamma)
        print ('--------------------------------------------------------------')
        
    # print zero-vector warning
    def print_zero_vector(self):
        print ('--------------------------WARNING-----------------------------')
        print ('x becomes zero vector. The solution may be incorrect.')
        print ('Current prune_gamma = %g, and Current epsilon = %g' % \
                (self.prune_gamma, self.epsilon))
        print ('Try smaller values of prune_gamma and epsilon or normalize y')
        print ('--------------------------------------------------------------')
        

# BSBL-FM : fast marginalized bsbl algos
class fm:
    
        # constructor
    def __init__(self, learn_lambda=2, y_scale=0.9,
                  epsilon=1e-8, max_iters=500, verbose=0,
                  learn_type=1, prune_gamma=1e-2):
        self.learn_lambda = learn_lambda
        self.y_scale = y_scale
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.verbose = verbose
        self.learn_type = learn_type
        self.prune_gamma = prune_gamma
        self.lamb = y_scale * 1e-2
        #
        if verbose:
            self.print_vars()
    
    
