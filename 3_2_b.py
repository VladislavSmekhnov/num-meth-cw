import numpy as np
from scipy.sparse import csc_matrix
import utils
import matplotlib.pyplot as plt
from math import *


def solve(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int = None, x0: np.ndarray = None):
    n = len(b)
    x0 = np.zeros(n) if x0 is None else x0
    max_iter = 100 * n if max_iter is None else max_iter

    A_l = A.copy()
    A_l = A_l.toarray().tolist()
    x = x0.copy()
    x = x.tolist()

    r = [0] * n
    p = [0] * n
    for i in range(n):
        r[i] = b[i] - utils.prod(A_l[i], x)
        p[i] = r[i]
    r_dot_r = utils.prod(r, r)
    count = 0
    r_list = []
    r_list.append(sqrt(utils.prod(r, r)))

    while r_dot_r > tol ** 2 and count < max_iter:
        Ap = [0] * n
        for i in range(n):
            Ap[i] = utils.prod(A_l[i], p)
        alpha = r_dot_r / utils.prod(p, Ap)
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        r_dot_r_new = utils.prod(r, r)
        beta = r_dot_r_new / r_dot_r
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        r_dot_r = r_dot_r_new
        count += 1
        r_list.append(sqrt(utils.prod(r, r)))

    return [x, count, r_list]


if __name__ == "__main__":
    a = 0
    b = 3
    N = 10
    h = (b - a) / N
    x_i = np.linspace(a, b, N)

    exact = []
    for i in x_i:
        exact.append(utils.exactSolution(i))

    linear_solve = utils.solution(x_i, h)

    A = utils.generate_tridiagonal_matrix(linear_solve[0], linear_solve[1], linear_solve[2])

    xi_ = x_i.copy()
    res = solve(A=A, b=linear_solve[3], tol=1e-6)
    sol = res[0]

    error = []
    for i in range(len(sol)):
        error.append(abs(exact[i] - sol[i]))

    plt.figure(figsize=(9, 6.75))
    plt.plot(x_i, exact, label='Точное решение')
    plt.plot(x_i, sol, label='Приближенное решение')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.xlim(a, b)
    plt.tight_layout()
    plt.show()

    print("{:<20} | {:<25} | {:<20}".format("Exact Solution", "Approximate Solution", "Error"))
    print("-" * 75)

    for ex, ap, er in zip(exact, sol, error):
        print("{:<20} | {:<25} | {:<20}".format(ex, ap, er))

    print("\nMax Error: {}".format(max(error)))
