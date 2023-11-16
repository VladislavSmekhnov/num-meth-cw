import numpy as np
from scipy.sparse import csc_matrix
import utils
import matplotlib.pyplot as plt
from math import *


def solve(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int = None, x0: np.ndarray = None):
    n = len(b)
    x0 = [0] * n if x0 is None else x0
    max_iter = 100 * n if max_iter is None else max_iter

    r_list = []
    rk = np.array(A * x0 - b, np.float128)
    r_list.append(np.sqrt(rk * rk))
    tk = sum(rk * rk) / sum((A.dot(rk)) * rk)
    x1 = x0 - tk * rk
    count = 0

    while count <= max_iter and sqrt(sum(rk * rk)) > tol * sqrt(sum(b * b)):
        x0 = x1
        rk = np.array(A * x0 - b, np.float128)
        r_list.append(rk)
        tk = sum(rk * rk) / sum((A.dot(rk)) * rk)
        x1 = x0 - tk * rk
        count += 1

    return [x1, count, r_list]


if __name__ == "__main__":
    a = 0
    b = 3
    N = 100
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
