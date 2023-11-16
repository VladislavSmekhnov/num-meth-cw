import math
import numpy as np
from scipy.sparse import diags


def exactSolution(x):
    return float(1) / ((float(1) + x) ** float(2))


def solution(x, h):
    A = np.zeros(len(x))
    B = np.zeros(len(x))
    C = np.zeros(len(x))
    D = np.zeros(len(x))

    A[0] = 0
    B[0] = (2 + math.pow(h, 2) * q(x[0]))
    C[0] = -1
    D[0] = exactSolution(x[0]) + math.pow(h, 2) * f(x[0])

    A[-1] = -1
    B[-1] = (2 + math.pow(h, 2) * q(x[-1]))
    C[-1] = 0
    D[-1] = exactSolution(x[-1]) + math.pow(h, 2) * f(x[-1])

    for i in range(1, len(x) - 1):
        A[i] = -1
        B[i] = (2 + math.pow(h, 2) * q(x[i]))
        C[i] = -1
        D[i] = math.pow(h, 2) * f(x[i])

    return A, B, C, D


def f(x):
    return 1 - 6 / ((1 + x) ** 4)


def q(x):
    return (1 + x) ** 2


def generate_tridiagonal_matrix(av, bv, cv):
    return diags([av[:-1], bv, cv[1:]], [-1, 0, 1], format="csc", dtype=np.float64)


def calc_r(A, x, b):
    n = len(x)
    Ax = np.zeros(n)
    for i in range(n):
        Ax[i] = np.dot(A[i], x)
    return Ax - b

# def fast_desc(av, bv, cv, fv, tol=1e-6, x0=None, maxiter=None):
#     n = len(fv)
#     x0 = [0] * n if x0 is None else x0
#     maxiter = 10 * n if maxiter is None else maxiter
#     A = [[0] * n for _ in range(n)]
#     for i in range(n):
#         A[i][i] = bv[i]
#         if i < n - 1:
#             A[i + 1][i] = av[i + 1]
#             A[i][i + 1] = cv[i]
#     r_list = []
#     r = calc_r(A, x0, fv)
#     r_list.append(sqrt(prod(r, r)))
#     tau = prod(r, r) / prod(prodMatrix(A, r), r)
#     x1 = [x0[i] - tau * r[i] for i in range(n)]
#     count = 0
#     while count <= maxiter and sqrt(prod(r, r)) > tol * sqrt(prod(fv, fv)):
#         x0 = x1
#         r = calc_r(A, x0, fv)
#         r_list.append(r)
#         tau = prod(r, r) / prod(prodMatrix(A, r), r)
#         x1 = [x0[i] - tau * r[i] for i in range(n)]
#
#     return [x1, count, r_list]


def prod(v1, v2):
    return sum([v1[i]*v2[i] for i in range(len(v1))])
