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

# print parameters
def print_vars(clf):
    print ('----------------------------INFO------------------------------')
    print ('apply lambda learning rule (learn_lambda) = %d' % clf.learn_lambda)
    print ('initial guess of noise      (lambda_init) = %g' % clf.lamb)
    print ('BSBL algorithm exit criterion   (epsilon) = %g' % clf.epsilon)
    print ('BSBL maximum iterations       (max_iters) = %d' % clf.max_iters)
    print ('intra-block correlation      (learn_type) = %d' % clf.learn_type)
    print ('Gamma pruning rules         (prune_gamma) = %g' % clf.prune_gamma)
    print ('--------------------------------------------------------------')

# vector to column 2D vector
def v2m(v):
    return v.reshape((v.shape[0],1))

# M = A*B*C
def dot3(A,B,C):
    return np.dot(np.dot(A,B), C)

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
def coeff_r(Cov, gamma, index, scale=1.1, r_init=0.90, r_thd=0.999):
    r0 = 0.
    r1 = 0.
    for i in index:
        temp = Cov[i] / gamma[i]
        #r0 += temp.diagonal().mean()
        #r1 += temp.diagonal(offset=1).mean()
        r0 += temp.trace()
        r1 += temp.trace(offset=1)
    # this method tend to under estimate the correlation
    if np.size(index) == 0:
        r = r_init
    else:
        r = scale * r1/(r0 + 1e-8)
    #
    if (np.abs(r) >= r_thd):
        r = r_thd * np.sign(r)
    return r

# generate toeplitz matrix
def gen_toeplitz(r,l):
    jup = np.arange(l)
    bs = r**jup
    B = lp.toeplitz(bs)
    return B

#
class bo:
    """
    BSBL-BO : Bound Optimization Algos of BSBL framework

    Recover block sparse signal (1D) exploiting intra-block correlation, 
    given the block partition.

    The algorithm solves the inverse problem for the block sparse
                model with known block partition:
                         y = X * w + v
    Variables
    ---------
    X : array, shape = (n_samples, n_features)
          Training vectors.

    y : array, shape = (n_samples)
        Target values for training vectors

    w : array, shape = (n_features)
        sparse/block sparse weight vector

    Parameters
    ----------
    'learn_lambda' : (1) if (SNR<10dB), learn_lambda=1
                     (2) if (SNR>10dB), learn_lambda=2
                     (3) if noiseless, learn_lambda=0
                     [ Default value: learn_lambda=2 ]
    'lambda_init'  : initial guess of the noise variance
                     [ Default value: lambda_init=1e-2 ]
    'r_init'       : initial value for correlation coefficient
                     [ Default value: 0.90 ]
    'epsilon'      : convergence criterion
    'max_iters'    : Maximum number of iterations.
                     [ Default value: max_iters = 500 ]
    'verbose'      : print debuging information
    'prune_gamma'  : threshold to prune out small gamma_i
                     (generally, 10^{-3} or 10^{-2})
    'learn_type'   : learn_type = 0: Ignore intra-block correlation
                     learn_type = 1: Exploit intra-block correlation
                     [ Default: learn_type = 1 ]
    """

    # constructor
    def __init__(self, learn_lambda=2, lambda_init=1e-2, r_init=0.90,
                  epsilon=1e-8, max_iters=500, verbose=0,
                  learn_type=1, prune_gamma=1e-2):
        self.learn_lambda = learn_lambda
        self.lamb = lambda_init
        self.r_init = r_init
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.verbose = verbose
        self.learn_type = learn_type
        self.prune_gamma = prune_gamma

    # fit y
    def fit_transform(self, X, y, blk_start_loc=None):
        #
        self.scale = y.std()
        y = y / self.scale
        #
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc==None:
            blkLen = int(N/16.)
            blk_start_loc = np.arange(0,N,blkLen)
        blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        # init variables
        nblock      = blk_start_loc.shape[0]
        w           = np.zeros(N,dtype='float')
        Sigma0      = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Sigma_w     = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Cov_x       = [np.identity(blk_len_list[i]) for i in range(nblock)]
        B           = [np.identity(blk_len_list[i]) for i in range(nblock)]
        invB        = [np.identity(blk_len_list[i]) for i in range(nblock)]
        block_slice = np.array([blk_start_loc[i] + np.arange(blk_len_list[i]) for i in xrange(nblock)])
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
                r = coeff_r(Cov_x, gamma, index, r_init=self.r_init)
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
                break
            if (count >= self.max_iters):
                break
        # exit
        self.count = count + 1
        self.gamma = gamma
        self.index = index
        # let's convert the backyard:
        w_ret = np.zeros(N)
        relevant_slice = ravel_list(block_slice[index])
        w_ret[relevant_slice] = w[relevant_slice]
        return w_ret * self.scale

    # print zero-vector warning
    def print_zero_vector(self):
        print ('--------------------------WARNING-----------------------------')
        print ('x becomes zero vector. The solution may be incorrect.')
        print ('Current prune_gamma = %g, and Current epsilon = %g' % \
                (self.prune_gamma, self.epsilon))
        print ('Try smaller values of prune_gamma and epsilon or normalize y')
        print ('--------------------------------------------------------------')

# compute logobj function for BSBL-FM
def logobj(s,q,A,L):
    As = np.dot(A, s)
    Aq = np.dot(A, q)
    ml = np.log(np.abs(lp.det(np.identity(L) + As))) - \
         np.dot(np.dot(q.T.conj(), lp.inv(np.identity(L) + As)), Aq)
    return ml

# extract the ith block index from current basis
def extract_segment(idx, basis_book, blk_len_list):
    N = sum(blk_len_list[basis_book])
    istart = 0
    for i in basis_book:
        if (i == idx):
            seg = np.arange(istart, istart+blk_len_list[i])
            break;
        istart += blk_len_list[i]
    #
    seg_others = np.ones(N, dtype='bool')
    seg_others[seg] = False
    return seg, seg_others

#
class fm:
    """
    BSBL-FM : fast marginalized bsbl algos

    Recover block sparse signal (1D) exploiting intra-block correlation, 
    given the block partition.

    The algorithm solves the inverse problem for the block sparse
                model with known block partition:
                         y = X * w + v

    Variables
    ---------
    X : array, shape = (n_samples, n_features)
        Training vectors.

    y : array, shape = (n_samples)
        Target values for training vectors

    w : array, shape = (n_features)
        sparse/block sparse weight vector

    Parameters
    ----------
    'learn_lambda' : (1) if (SNR<10dB), learn_lambda=1
                     (2) if (SNR>10dB), learn_lambda=2
                     (3) if noiseless, learn_lambda=0
                     [ Default value: learn_lambda=2 ]
    'lambda_init'  : initial guess of the noise variance
                     [ Default value: lambda_init=1e-2 ]
    'r_init'       : initial value for correlation coefficient
                     [ Default value: 0.90 ]
    'epsilon'      : convergence criterion
    'max_iters'    : Maximum number of iterations.
                     [ Default value: max_iters = 500 ]
    'verbose'      : print debuging information
    'prune_gamma'  : threshold to prune out small gamma_i
                     (generally, 10^{-3} or 10^{-2})
    'learn_type'   : learn_type = 0: Ignore intra-block correlation
                     learn_type = 1: Exploit intra-block correlation 
                     [ Default: learn_type = 1 ]
    """

    # constructor
    def __init__(self, learn_lambda=2, r_init=0.90, lambda_init=1e-2,
                  epsilon=1e-4, max_iters=500, verbose=0,
                  learn_type=1, prune_gamma=1e-2):
        self.learn_lambda = learn_lambda
        self.lamb = lambda_init
        self.r_init = r_init
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.verbose = verbose
        self.learn_type = learn_type
        self.prune_gamma = prune_gamma

    # fit y
    def fit_transform(self, X, y, blk_start_loc=None):
        # normalize y
        self.scale = y.std()
        y = y / self.scale
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc==None:
            blkLen = int(N/16.)
            blk_start_loc = np.arange(0,N,blkLen)
        blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        #
        # init variables
        #
        nblock      = blk_start_loc.shape[0]
        self.nblock = nblock
        block_slice = np.array([blk_start_loc[i] + np.arange(blk_len_list[i]) for i in range(nblock)])
        Xs          = [X[:,block_slice[i]] for i in range(nblock)]
        beta        = 1. / self.lamb
        S           = [np.dot(beta*Xs[i].T.conj(), Xs[i]) for i in range(nblock)]
        Q           = [np.dot(beta*Xs[i].T.conj(), y) for i in range(nblock)]
        # index is the 1/0 indicator for relevant block-basis
        index       = np.zeros(nblock, dtype='bool')
        Am          = [np.zeros(S[i].shape) for i in range(nblock)]
        self.gamma  = np.zeros(nblock, dtype='float')
        #
        ml, A, theta = self.logobj_mapping(index, S, Q, Am)
        idx = ml.argmin(0)
        if self.verbose: print ('bsbl-fm bootup, add %d' % idx)
        # add this basis
        index[idx] = True
        Am[idx] = A[idx]
        self.gamma[idx] = lp.norm(Am[idx])
        basis_book = idx
        # initial {Sig, w}
        Sigma_ii = np.dot(lp.inv(np.identity(blk_len_list[idx]) + np.dot(Am[idx], S[idx])), Am[idx])
        Sig      = Sigma_ii
        w        = np.dot(Sigma_ii, Q[idx])
        Xu       = Xs[idx]
        # update {S, Q}
        for k in range(nblock):
            Xk    = Xs[k]
            XSX   = dot3(Xu, Sig, Xu.T.conj())
            S[k]  = S[k] - beta**2*dot3(Xk.T.conj(), XSX, Xk)
            Q[k]  = Q[k] - beta*dot3(Xk.T.conj(), Xu, w)
        # loops
        ML = np.zeros(self.max_iters)
        for count in range(1,self.max_iters):
            ml, A, theta = self.logobj_mapping(index, S, Q, Am)
            idx = ml.argmin(0)
            #
            # check convergence now
            ML[count] = ml[idx]
            if (ML[count] >= 0):
                break
            if count>1:
                ml_ratio = np.abs(ML[count] - ML[count-1]) / np.abs(ML[count] - ML[0])
                if ml_ratio < self.epsilon:
                    break
            #
            # operation on basis
            if index[idx]==True:
                seg, segc = extract_segment(idx, basis_book, blk_len_list)
                Sig_j = Sig[:,seg]
                Sig_jj = Sig[seg,:][:,seg]
                if theta[idx] > self.prune_gamma:
                    if self.verbose: print ('re-estimate %d' % idx)
                    Xu = X[:, ravel_list(block_slice[basis_book])]
                    #
                    Denom = lp.inv(Sig_jj + np.dot(np.dot(Am[idx], lp.inv(Am[idx] - A[idx])), A[idx]))
                    ki = dot3(Sig_j, Denom, Sig_j.T.conj())
                    Sig = Sig - ki
                    w = w - beta*dot3(ki,Xu.T.conj(), y)
                    PKP = dot3(Xu, ki, Xu.T.conj())
                    for k in range(nblock):
                        Phi_m = Xs[k];
                        PPKP = np.dot(Phi_m.T.conj(), PKP)
                        S[k] = S[k] + beta**2*np.dot(PPKP, Phi_m)
                        Q[k] = Q[k] + beta**2*np.dot(PPKP, y)
                    Am[idx] = A[idx]
                    self.gamma[idx] = lp.norm(Am[idx])
                else:
                    if self.verbose: print ('delete %d' % idx)
                    Am[idx] = np.zeros(Am[idx].shape)
                    self.gamma[idx] = 0.
                    index[idx] = False
                    #
                    ki = dot3(Sig_j, lp.inv(Sig_jj), Sig_j.T.conj())
                    Sig = Sig - ki;
                    w = w - beta*dot3(ki, Xu.T.conj(), y)
                    PKP = dot3(Xu, ki, Xu.T.conj())
                    for k in range(nblock):
                        Phi_m = Xs[k]
                        PPKP = np.dot(Phi_m.T.conj(), PKP)
                        S[k] = S[k] + beta**2*np.dot(PPKP, Phi_m)
                        Q[k] = Q[k] + beta**2*np.dot(PPKP, y)
                    #
                    w = w[segc]
                    Sig = Sig[:,segc][segc,:]
                    basis_book = np.delete(basis_book, np.argwhere(basis_book==idx))
                    Xu = X[:, ravel_list(block_slice[basis_book])]
            else:
                if self.verbose: print ('add %d' % idx)
                Am[idx] = A[idx]
                self.gamma[idx] = lp.norm(Am[idx])
                index[idx] = True
                Xk = Xs[idx]
                #
                Sigma_ii = np.dot(lp.inv(np.identity(blk_len_list[idx]) + np.dot(A[idx], S[idx])), A[idx])
                mu_i = np.dot(Sigma_ii, Q[idx])
                SPP = np.dot(np.dot(Sig, Xu.T.conj()), Xk)
                Sigma_11 = Sig + beta**2*np.dot(np.dot(SPP, Sigma_ii), SPP.T.conj())
                Sigma_12 = -beta*np.dot(SPP, Sigma_ii)
                Sigma_21 = Sigma_12.T.conj()
                #
                mu_1 = w - beta*np.dot(SPP, mu_i)
                e_i = Xk - beta*np.dot(Xu, SPP)
                ESE = np.dot(np.dot(e_i, Sigma_ii), e_i.T.conj())
                for k in range(nblock):
                    Phi_m = Xs[k]
                    S[k] = S[k] - beta**2*np.dot(np.dot(Phi_m.T.conj(), ESE), Phi_m)
                    Q[k] = Q[k] - beta*np.dot(np.dot(Phi_m.T.conj(), e_i), mu_i)
                # adding relevant basis
                Sig = np.vstack((np.hstack((Sigma_11,Sigma_12)),  \
                                 np.hstack((Sigma_21,Sigma_ii))))
                w = np.r_[mu_1,mu_i]
                basis_book = np.append(basis_book, idx)
                Xu = X[:, ravel_list(block_slice[basis_book])]

        # store the ||A||_F norm as the gammas
        self.count = count
        # exit
        w_ret = np.zeros(N)
        relevant_slice = ravel_list(block_slice[basis_book])
        w_ret[relevant_slice] = w
        return w_ret * self.scale

    #
    #
    def logobj_mapping(self, index, S, Q, Am):
        N = self.nblock
        s = S
        q = Q
        for i in np.argwhere(index):
            invDenom = lp.inv(np.identity(Am[i].shape[0]) - S[i]*Am[i])
            s[i] = np.dot(invDenom, S[i])
            q[i] = np.dot(invDenom, Q[i])
        #
        theta = np.zeros(N)
        A = [np.zeros(S[i].shape) for i in range(N)]
        for i in range(N):
            #invSK = lp.inv(s[i])
            #invSK = np.diag(1./np.diag(s[i]))
            #A[i] = dot3(invSK, (np.dot(q[i],q[i].T.conj()) - s[i]), invSK)
            sq = np.dot(s[i], q[i])
            A[i] = np.dot(sq, sq.T.conj()) - lp.inv(s[i])
            theta[i] = 1.0/A[i].shape[0] * np.real(A[i].trace())
        # learn
        if self.learn_type == 1:
            r = coeff_r(Am, self.gamma, np.argwhere(index), r_init=self.r_init)
            if self.is_equal_block:
                Bc = gen_toeplitz(r, A[0].shape[0])
                A = [Bc for i in range(N)]
            else:
                A = [gen_toeplitz(r, A[i].shape[0]) for i in range(N)]
        else:
            A = [np.identity(A[i].shape[0]) * theta[i] for i in range(N)]

        #
        candidate_new = theta > self.prune_gamma
        candidate_add = candidate_new & (~index)
        candidate_del = (~candidate_new) & index
        candidate_est = candidate_new & index
        # init
        ml = np.inf * np.ones(theta.size, dtype='float')
        # add
        for i in np.argwhere(candidate_add):
            ml[i] = logobj(s[i], q[i], A[i], A[i].shape[0])
        # del
        for i in np.argwhere(candidate_del):
            ml[i] = -logobj(s[i], q[i], A[i], A[i].shape[0])
        # re-estimate
        for i in np.argwhere(candidate_est):
            ml[i] = logobj(s[i], q[i], A[i], A[i].shape[0]) - \
                    logobj(s[i], q[i], Am[i], Am[i].shape[0])
        return ml, A, theta

    #
    def add(self):
        return 1

    #
    def delete(self):
        return 1

    #
    def estimate(self):
        return 1
