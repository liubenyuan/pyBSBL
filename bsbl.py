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
# @article{zhang2013extension,
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
# @article{liu2014energy,
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
# @article{zhang2013compressed,
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
    print("----------------------------INFO------------------------------")
    print("apply lambda learning rule (learn_lambda) = %d" % clf.learn_lambda)
    print("initial guess of noise      (lambda_init) = %g" % clf.lamb)
    print("BSBL algorithm exit criterion   (epsilon) = %g" % clf.epsilon)
    print("BSBL maximum iterations       (max_iters) = %d" % clf.max_iters)
    print("intra-block correlation      (learn_type) = %d" % clf.learn_type)
    print("Gamma pruning rules         (prune_gamma) = %g" % clf.prune_gamma)
    print("--------------------------------------------------------------")


# vector to column (M,1) vector
def v2m(v):
    return v.reshape((v.shape[0], 1))


# M = A*B*C
def dot3(A, B, C):
    return np.dot(np.dot(A, B), C)


# ravel list of 'unequal arrays' into a row vector
def ravel_list(d):
    r = np.array([], dtype="int")
    for i in range(d.shape[0]):
        r = np.r_[r, d[i]]
    return r


# extract block spacing information
def block_parse(blk_start_loc, N):
    blk_len_list = np.r_[blk_start_loc[1:], N] - blk_start_loc
    is_equal_block = np.sum(np.abs(blk_len_list - blk_len_list.mean())) == 0
    return blk_len_list, is_equal_block


# exploit AR(1) correlation in Covariance matrices
#   r_scale : scale the estimated coefficient
#   r_init : initial guess of r when no-basis is included
#   r_thd : the threshold of r to make the covariance matrix p.s.d
#           the larger the block, the smaller the value
def coeff_r(Cov, gamma, index, r_scale=1.1, r_init=0.90, r_thd=0.999):
    r0 = 0.0
    r1 = 0.0
    for i in index:
        temp = Cov[i] / gamma[i]
        r0 += temp.trace()
        r1 += temp.trace(offset=1)
    # this method tend to under estimate the correlation
    if np.size(index) == 0:
        r = r_init
    else:
        r = r_scale * r1 / (r0 + 1e-8)
    # constrain the Toeplitz matrix to be p.s.d
    if np.abs(r) >= r_thd:
        r = r_thd * np.sign(r)
    return r


# generate toeplitz matrix
def gen_toeplitz(r, k):
    jup = np.arange(k)
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
    def __init__(
        self,
        learn_lambda=2,
        lambda_init=1e-2,
        r_init=0.90,
        epsilon=1e-8,
        max_iters=500,
        verbose=0,
        learn_type=1,
        prune_gamma=1e-2,
    ):
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
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc is None:
            blkLen = int(N / 16.0)
            blk_start_loc = np.arange(0, N, blkLen)
        blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        # init variables
        nblock = blk_start_loc.shape[0]
        self.nblock = nblock
        w = np.zeros(N, dtype="float")
        Sigma0 = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Sigma_w = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Cov_x = [np.identity(blk_len_list[i]) for i in range(nblock)]
        B = [np.identity(blk_len_list[i]) for i in range(nblock)]
        invB = [np.identity(blk_len_list[i]) for i in range(nblock)]
        block_slice = np.array(
            [blk_start_loc[i] + np.arange(blk_len_list[i]) for i in range(nblock)]
        )
        gamma = np.ones(nblock, dtype="float")
        HX = [np.identity(blk_len_list[i]) for i in range(nblock)]
        Hy = [np.zeros(blk_len_list[i]) for i in range(nblock)]
        # loops
        for count in range(self.max_iters):
            # prune weights as their hyperparameter goes to zero
            # index -- 0:unused, 1:used
            index = np.argwhere(gamma > self.prune_gamma).ravel()
            if index.shape[0] == 0:
                self.print_zero_vector()
                raise TypeError("w is a zero-vector, exiting.")
            # calculate XBX^T
            XBX = np.zeros((M, M), dtype=float)
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

            # =========== Learn correlation structure in blocks ===========
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
                lambComp = 0.0
                for i in index:
                    Xi = X[:, block_slice[i]]
                    lambComp += np.dot(np.dot(Xi, Sigma_w[i]), Xi.T).trace()
                self.lamb = lp.norm(y - np.dot(X, w)) ** 2.0 / N + lambComp / N
            elif self.learn_lambda == 2:
                lambComp = 0.0
                for i in index:
                    lambComp += np.dot(Sigma_w[i], invB[i]).trace() / gamma_old[i]
                self.lamb = (
                    lp.norm(y - np.dot(X, w)) ** 2.0 / N
                    + self.lamb * (w.size - lambComp) / N
                )

            # ================= Check stopping conditions, eyc. ==============
            dmu = (np.abs(w_old - w)).max(0)
            # only SMV currently
            if dmu < self.epsilon:
                break
            if count >= self.max_iters:
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
        print("--------------------------WARNING-----------------------------")
        print("x becomes zero vector. The solution may be incorrect.")
        print(
            "Current prune_gamma = %g, and Current epsilon = %g"
            % (self.prune_gamma, self.epsilon)
        )
        print("Try smaller values of prune_gamma and epsilon or normalize y")
        print("--------------------------------------------------------------")


#
# compute logobj cost likelihood for BSBL-FM
#   L(i) = log(|I + A_is_i|) - q_i^T(I + A_is_i)^{-1}A_iq_i
def logobj(s, q, A, L):
    As = np.dot(A, s)
    Aq = np.dot(A, q)
    ml = np.log(np.abs(lp.det(np.identity(L) + As))) - dot3(
        q.T.conj(), lp.inv(np.identity(L) + As), Aq
    )
    return ml


# calculate Sigma_ii:
#   \Sigma_{ii} = (A^{-1} + S)^{-1} = (I + AS)^{-1}*A
def calc_sigmaii(A, S):
    L = A.shape[0]
    return np.dot(lp.inv(np.eye(L) + np.dot(A, S)), A)


# extract the ith block index 'within' current basis
def extract_segment(idx, basis_book, blk_len_list):
    N = sum(blk_len_list[basis_book])
    istart = 0
    for i in basis_book:
        if i == idx:
            seg = np.arange(istart, istart + blk_len_list[i])
            break
        istart += blk_len_list[i]
    #
    seg_others = np.ones(N, dtype="bool")
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
    def __init__(
        self,
        learn_lambda=2,
        r_init=0.90,
        lambda_init=1e-2,
        epsilon=1e-4,
        max_iters=500,
        verbose=0,
        learn_type=1,
        prune_gamma=1e-2,
    ):
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
        """
        solve y = Xw + v, with block indices specified by blk_start_loc

        Parameters
        ----------
        X              : MxN np.array
        y              : M   np.array
        blk_start_loc  : block indices, [Optional]
                         if unspecified, it will uniformly devide v
                         into 16 blocks

        Output
        ------
        w              : N   np.array
        """
        # normalize y
        self.scale = y.std()
        y = y / self.scale
        M, N = X.shape
        # automatically set block partition
        if blk_start_loc is None:
            blkLen = int(N / 16.0)
            blk_start_loc = np.arange(0, N, blkLen)
        self.blk_len_list, self.is_equal_block = block_parse(blk_start_loc, N)
        # init variables
        self.init(X, y, blk_start_loc)
        # bootstrap ADD one basis
        ml, A, theta = self.logobj_mapping()
        idx = ml.argmin(0)
        Sig, w, Xu = self.bootup(A, idx)
        # loops
        ML = np.zeros(self.max_iters)
        ML[0] = ml[idx]
        for count in range(1, self.max_iters):
            ml, A, theta = self.logobj_mapping()
            idx = ml.argmin(0)

            # check convergence now
            ML[count] = ml[idx]
            if ML[count] >= 0:
                break
            if count > 1:
                ml_ratio = np.abs(ML[count] - ML[count - 1]) / np.abs(ML[count] - ML[0])
                if ml_ratio < self.epsilon:
                    break

            # operation on basis
            if self.index[idx] is True:
                if theta[idx] > self.prune_gamma:
                    proc = self.estimate
                else:
                    proc = self.delete
            else:
                proc = self.add
            # process Sig, w, Xu
            Sig, w, Xu = proc(Sig, w, Xu, A, idx)

        # exit
        self.count = count
        return self.w_format(w)

    # initialize quantiles
    def init(self, X, y, blk_start):
        blk_len = self.blk_len_list
        nblock = blk_start.shape[0]
        beta = 1.0 / self.lamb
        block_slice = [blk_start[i] + np.arange(blk_len[i]) for i in range(nblock)]
        Xs = [X[:, block_slice[i]] for i in range(nblock)]
        # init {S,Q}
        self.S = [np.dot(beta * Xs[i].T.conj(), Xs[i]) for i in range(nblock)]
        self.Q = [np.dot(beta * Xs[i].T.conj(), y) for i in range(nblock)]
        # store {X, slice}
        self.slice = np.array(block_slice)
        self.Xs = Xs
        # index is the 1/0 indicator for relevant block-basis
        self.index = np.zeros(nblock, dtype="bool")
        self.Am = [np.zeros((blk_len[i], blk_len[i])) for i in range(nblock)]
        self.gamma = np.zeros(nblock, dtype="float")
        # store {y}
        self.y = y
        self.nblock = nblock
        self.beta = beta

    #
    def logobj_mapping(self):
        N = self.nblock
        index = self.index
        S = self.S
        Q = self.Q
        Am = self.Am
        #
        s = S
        q = Q
        for i in np.argwhere(index):
            invDenom = lp.inv(np.identity(Am[i].shape[0]) - S[i] * Am[i])
            s[i] = np.dot(invDenom, S[i])
            q[i] = np.dot(invDenom, Q[i])
        #
        theta = np.zeros(N)
        A = [np.zeros(S[i].shape) for i in range(N)]
        for i in range(N):
            # invSK = lp.inv(s[i])
            # invSK = np.diag(1./np.diag(s[i]))
            # A[i] = dot3(invSK, (np.dot(q[i],q[i].T.conj()) - s[i]), invSK)
            sq = np.dot(s[i], q[i])
            A[i] = np.dot(sq, sq.T.conj()) - lp.inv(s[i])
            theta[i] = 1.0 / A[i].shape[0] * np.real(A[i].trace())
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
        ml = np.inf * np.ones(theta.size, dtype="float")
        # add
        for i in np.argwhere(candidate_add):
            ml[i] = logobj(s[i], q[i], A[i], A[i].shape[0])
        # del
        for i in np.argwhere(candidate_del):
            ml[i] = -logobj(s[i], q[i], A[i], A[i].shape[0])
        # re-estimate
        for i in np.argwhere(candidate_est):
            ml[i] = logobj(s[i], q[i], A[i], A[i].shape[0]) - logobj(
                s[i], q[i], Am[i], Am[i].shape[0]
            )
        return ml, A, theta

    #
    def bootup(self, A, idx):
        if self.verbose:
            print("bsbl-fm bootup, add %d" % idx)
        #
        self.index[idx] = True
        self.Am[idx] = A[idx]
        self.gamma[idx] = lp.norm(A[idx])
        self.basis_book = idx
        # initial {Sig, w}
        Sigma_ii = calc_sigmaii(A[idx], self.S[idx])
        Sig = Sigma_ii
        w = np.dot(Sigma_ii, self.Q[idx])
        Xu = self.Xs[idx]
        XSX = dot3(Xu, Sig, Xu.T.conj())
        # update {S, Q}
        for k in range(self.nblock):
            Xk = self.Xs[k]
            self.S[k] = self.S[k] - self.beta**2 * dot3(Xk.T.conj(), XSX, Xk)
            self.Q[k] = self.Q[k] - self.beta * dot3(Xk.T.conj(), Xu, w)
        #
        return Sig, w, Xu

    #
    def add(self, Sig, w, Xu, A, idx):
        if self.verbose:
            print("add %d" % idx)
        #
        Xi = self.Xs[idx]
        Sigma_ii = calc_sigmaii(A[idx], self.S[idx])
        mu_i = np.dot(Sigma_ii, self.Q[idx])
        # update Sig
        SPP = np.dot(np.dot(Sig, Xu.T.conj()), Xi)
        Sigma_11 = Sig + self.beta**2 * dot3(SPP, Sigma_ii, SPP.T.conj())
        Sigma_12 = -self.beta * np.dot(SPP, Sigma_ii)
        Sigma_21 = Sigma_12.T.conj()
        Sig = np.vstack(
            (np.hstack((Sigma_11, Sigma_12)), np.hstack((Sigma_21, Sigma_ii)))
        )
        # update w
        mu = w - self.beta * np.dot(SPP, mu_i)
        w = np.r_[mu, mu_i]
        # update {S, Q}
        e_i = Xi - self.beta * np.dot(Xu, SPP)
        ESE = dot3(e_i, Sigma_ii, e_i.T.conj())
        for k in range(self.nblock):
            Xk = self.Xs[k]
            self.S[k] = self.S[k] - self.beta**2 * dot3(Xk.T.conj(), ESE, Xk)
            self.Q[k] = self.Q[k] - self.beta * dot3(Xk.T.conj(), e_i, mu_i)
        # adding relevant basis
        self.Am[idx] = A[idx]
        self.gamma[idx] = lp.norm(A[idx])
        self.index[idx] = True
        self.basis_book = np.append(self.basis_book, idx)
        Xu = np.c_[Xu, Xi]
        return Sig, w, Xu

    #
    def delete(self, Sig, w, Xu, A, idx):
        if self.verbose:
            print("delete %d" % idx)
        #
        basis_book = self.basis_book
        seg, segc = extract_segment(idx, basis_book, self.blk_len_list)
        print(basis_book)
        print(sum(self.blk_len_list[basis_book]))
        Sig_j = Sig[:, seg]
        Sig_jj = Sig[seg, :][:, seg]
        # del
        ki = dot3(Sig_j, lp.inv(Sig_jj), Sig_j.T.conj())
        Sig = Sig - ki
        print(w.shape)
        w = w - self.beta * dot3(ki, Xu.T.conj(), self.y)
        XKX = dot3(Xu, ki, Xu.T.conj())
        for k in range(self.nblock):
            Xk = self.Xs[k]
            XXKX = np.dot(Xk.T.conj(), XKX)
            self.S[k] = self.S[k] + self.beta**2 * np.dot(XXKX, Xk)
            self.Q[k] = self.Q[k] + self.beta**2 * np.dot(XXKX, self.y)
        # delete
        print(w.shape)
        print(segc.shape)
        w = w[segc]
        Sig = Sig[:, segc][segc, :]
        Xu = Xu[:, segc]
        self.Am[idx] = np.zeros(self.Am[idx].shape)
        self.gamma[idx] = 0.0
        self.index[idx] = False
        self.basis_book = np.delete(basis_book, np.argwhere(basis_book == idx))
        return Sig, w, Xu

    #
    def estimate(self, Sig, w, Xu, A, idx):
        if self.verbose:
            print("re-estimate %d" % idx)
        #
        basis_book = self.basis_book
        seg, segc = extract_segment(idx, basis_book, self.blk_len_list)
        Sig_j = Sig[:, seg]
        Sig_jj = Sig[seg, :][:, seg]
        # reestimate
        Denom = lp.inv(
            Sig_jj + np.dot(np.dot(self.Am[idx], lp.inv(self.Am[idx] - A[idx])), A[idx])
        )
        ki = dot3(Sig_j, Denom, Sig_j.T.conj())
        Sig = Sig - ki
        w = w - self.beta * dot3(ki, Xu.T.conj(), self.y)
        XKX = dot3(Xu, ki, Xu.T.conj())
        for k in range(self.nblock):
            Xk = self.Xs[k]
            XXKX = np.dot(Xk.T.conj(), XKX)
            self.S[k] = self.S[k] + self.beta**2 * np.dot(XXKX, Xk)
            self.Q[k] = self.Q[k] + self.beta**2 * np.dot(XXKX, self.y)
        #
        self.Am[idx] = A[idx]
        self.gamma[idx] = lp.norm(A[idx])
        self.index[idx] = True
        return Sig, w, Xu

    # format block sparse w into w
    def w_format(self, w):
        w_ret = np.zeros(sum(self.blk_len_list))
        relevant_slice = ravel_list(self.slice[self.basis_book])
        w_ret[relevant_slice] = w
        return w_ret * self.scale
