import numpy as np
import utils
import matplotlib.pyplot as plt


def solve(a: np.array, b: np.array, c: np.array, f: np.array, omega: float, tol: float, x0: np.ndarray = None,
          max_iter: int = None):
    n = len(f)
    x0 = [0] * n if x0 is None else x0
    max_iter = 10 * n if max_iter is None else max_iter

    r_k = np.array([], dtype=np.float128)

    for _ in range(max_iter):
        x_new = np.zeros(len(x0), dtype=np.float128)
        diff = 0
        for i in range(len(x0)):
            x_new[i] = (1 - omega) * x0[i]
            x_right = f[i]
            if i > 0:
                x_right -= a[i] * x_new[i - 1]
            if i < len(x0) - 1:
                x_right -= c[i] * x0[i + 1]
            x_new[i] += omega * x_right / b[i]
            if abs(x_new[i] - x0[i]) > diff:
                diff = abs(x_new[i] - x0[i])

        x0 = x_new

        max_rk = 0
        for i in range(i):
            Ax = b[i] * x0[i]
            if i > 0:
                Ax += a[i] * x0[i - 1]
            if i < len(x0) - 1:
                Ax += c[i] * x0[i + 1]
            if abs(Ax - f[i]) > max_rk:
                max_rk = abs(Ax - f[i])
        r_k = np.append(r_k, max_rk)

        if diff <= tol:
            break
    # else:
    #     raise ValueError("Maximum number of iterations reached")
    return x0, _, r_k


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

    xi_ = x_i.copy()
    res = solve(a=linear_solve[0], b=linear_solve[1], c=linear_solve[2], f=linear_solve[3], tol=1e-6, x0=xi_, omega=1.7)
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
