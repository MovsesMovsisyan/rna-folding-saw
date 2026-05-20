from numba import njit, prange, f8
from numpy import zeros, linspace, exp, array, ones, log as ln
from numpy.random import random, choice
from numpy import savez_compressed as nsave
from numpy import arange, load as nload


@njit([f8[:](f8, f8)], nogil=True)
def triplet(x, y):
    return linspace(x - y, x + y, 3)


@njit(nogil=True)
def Q(beta, e0, e, alpha, h1, h2, i, j):
    hi, hj = 1, 1
    if h1:
        hi = -1
    if h2:
        hj = -1
    return exp(-beta * (e0 + e * hi * hj + alpha * (j - i)))


@njit(nogil=True)
def Partition(seq, N, beta, e0, e, alpha):
    f = ones((seq.shape[0], seq.shape[0]))
    for j in range(seq.shape[0]):
        for i in range(0, j):
            f[i][j] = f[i][j - 1]
            if i == j - 1:
                f[i][j] += Q(beta, e0, e, alpha, seq[i], seq[j], i, j)
            else:
                f[i][j] += Q(beta, e0, e, alpha, seq[i], seq[j], i, j) * f[i + 1][j - 1]
                f[i][j] += f[i][j - 2] * Q(beta, e0, e, alpha, seq[j - 1], seq[j], j - 1, j)
                for k in range(i + 1, j - 1):
                    f[i][j] += f[i][k - 1] * Q(beta, e0, e, alpha, seq[k], seq[j], k, j) * f[k + 1][j - 1]
    return -ln(f[0][N - 1]) / N


@njit(nogil=True)
def FreeEnergy(N, M, q, beta, e0, e, alpha):
    F = zeros((len(beta), len(e0)))
    for i in range(M):
        seq = random(N) > q
        for j in prange(len(beta)):
            for k in prange(len(e0)):
                F[j][k] += Partition(seq, N, beta[j], e0[k], e, alpha)
    return F / M


@njit(nogil=True)
def FreeEnergyGeometric(N, M, q, beta, e0, e, alpha, dl_sample):
    F = zeros((len(beta), len(e0)))
    S = dl_sample.shape[0]
    for i in range(M):
        seq = random(N) > q
        for j in prange(len(beta)):
            for k in range(len(e0)):
                acc = 0.0
                for s in range(S):
                    alpha_modified = alpha * dl_sample[s]
                    acc += Partition(seq, N, beta[j], e0[k], e, alpha_modified)
                F[j][k] += acc / S
    return F / M


q = 0.7
paramE = 0.5
paramS = 0
paramAlphas = array([0, 0.5, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0])
beta = arange(0.1, 3, 0.01) ** -1
mu = arange(-40, 0, 0.001)
e0 = triplet(-1, 0.01)
e = abs(e0[1]) * paramE

DL_ALL = nload("datafiles/dl_ratios.npy")
DL_NORM = DL_ALL / DL_ALL.mean()
N_TOTAL = DL_NORM.shape[0]     # number of D/(1-l) ratios in the dataset (200)
SUB = 50                       # subsample size per bootstrap iteration
B = 30                         # number of bootstrap iterations (independent subsamples)
M_SEQ = 1                      
N_CHAIN = 25

freesna = zeros((B, len(paramAlphas), len(beta), len(e0)))

for b in range(B):
    idx = choice(N_TOTAL, size=SUB, replace=False)
    sample = DL_NORM[idx]  
    for ai in range(len(paramAlphas)):
        freesna[b][ai] = FreeEnergyGeometric(
            N_CHAIN, M_SEQ, q, beta, e0, e,
            paramAlphas[ai], sample
        )
    print(f"bootstrap {b+1}/{B} done")

nsave("datafiles/seq_avg_1st_na", freesna)
