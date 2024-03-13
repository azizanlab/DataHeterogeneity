import numpy as np
import math
import matplotlib.pyplot as plt

step_size = 0.01 #step_size size for plotting the pdf/cdf.

def calculate_theta_h(A, m):
    '''
    :param A: A square nxn matrix.
    :param m: The number of machines we are distributing the system to. For simplicity the code requires that m divides n.
    :return: The value of cos(theta_h), where theta_h, as described in the paper, using Algorithm 1.
    '''
    n = A.shape[0]
    p = n//m
    Qs = []
    cos_theta_h = 0
    for i in range(m):
        Q, R = np.linalg.qr(A[p*i : p*(i + 1), :].T)
        Qs. append(Q)
    for i in range(m):
        for j in range(i + 1, m):
            Sigma = np.linalg.svd(Qs[i].T@Qs[j])[1]
            cos_theta_h = max(cos_theta_h, np.max(Sigma))
    return cos_theta_h


def partial_factorial(a, k):
    '''
    :param a: Starting number.
    :param k: Depth.
    :return: a(a - 1)...(a - k + 1).
    '''
    return (np.arange(k) + a).prod()


def partitional_shifted_factorial(a, kappa):
    '''
    :param a: The number.
    :param kappa: A partition
    :return: Hypergeometric coefficient (a)_kappa.
    '''
    res = 1
    for j, k in enumerate(kappa):
        res *= partial_factorial(a - j/2, k)
    return res


def zonal_pol_adjusted(kappa, k, p):
    '''
    :param kappa: Partition, all entries are positive.
    :param k: Sum of the partition (the number we are partitioning).
    :param p: Length of the partition when zeros are included. Hence len(kappa) <= p.
    :return: Zonal polynomial evaluated at I_p, with respect to the partition kappa of length p (padded with zeroes), divided by k!.
    '''
    fraction=1
    for j in range(1, len(kappa) + 1):
        denominator = math.factorial(2*kappa[j - 1] + len(kappa) - j)
        numerator = 1
        for l in range(j + 1, len(kappa) + 1):
            numerator *= (2*kappa[j - 1] - 2*kappa[l - 1] - j + l)
        fraction *= numerator/denominator
    return 2**(2*k)*partitional_shifted_factorial(p/2, kappa)*fraction


def partitions(n, l, I=1):
    '''
    :param n: The number we are partitioning.
    :param l: Maximum allowed length of the partition.
    :return: Yields all partitions of maximum length l of the number n.
    '''
    yield (n,)
    if l == 1: return
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, l - 1, i):
            yield p + (i,)


def get_cdf(n, p, left, right):
    '''
    :param n: The dimension of the space.
    :param p: The dimension of the subspace.
    :param left: Left bound for cdf calculation.
    :param right: Right bound for cdf cdlculation.
    :return: If theta is the smallest principal angle between two randomly selected p-dimensional subspaces of R^n,
    the cdf of (tan(theta_h))^(-2) is returned. Calculated using the formula as described in the paper.
    '''
    x = np.arange(start=left, stop=right, step=step_size)
    res = np.ones(x.shape[0])
    for k in range(1, p*(n - 2*p - 1)//2 + 1):
        for kappa in partitions(k, p):
            if kappa[0] > (n - 2*p - 1)//2: continue
            res += partitional_shifted_factorial(p/2, kappa)*zonal_pol_adjusted(kappa, k, p)/((1 + x)**k)
    return res*(x/(x + 1))**(p**2/2)


def get_pdf(n, p, left, right):
    '''
    :param n: The dimension of the space.
    :param p: The dimension of the subspace.
    :param left: Left bound for pdf calculation.
    :param right: Right bound for pdf cdlculation.
    :return: Gets the pdf from the cdf of the smallest principal angle using finite differences.
    '''
    cdf = get_cdf(n, p, left, right)
    return np.diff(cdf)/step_size
