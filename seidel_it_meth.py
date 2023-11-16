import numpy as np
import utils
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, f: np.ndarray, tol: float, max_iter: int=None,  x: np.ndarray=None)-> tuple[np.ndarray, int, np.ndarray]:
    n = len(f)
    x = [0] * n if x is None else x
    max_iter = 10 * n if max_iter is None else max_iter

    r_k = np.array([], dtype=np.float128)

    for _ in range(max_iter):
        diff = 0
        for i in range(len(x)):
            x_new = f[i]
            if i > 0:
                x_new -= a[i] * x[i - 1]
            if i < len(x) - 1:
                x_new -= c[i] * x[i + 1]
            x_new /= b[i]
            if abs(x_new - x[i]) > diff:
                diff = abs(x_new - x[i])
            x[i] = x_new

        max_rk = 0
        for i in range(i):
            Ax = b[i] * x[i]
            if i > 0:
                Ax += a[i] * x[i - 1]
            if i < len(x) - 1:
                Ax += c[i] * x[i + 1]
            if abs(Ax - f[i]) > max_rk:
                max_rk = abs(Ax - f[i])
        r_k = np.append(r_k, max_rk)

        if diff <= tol:
            break
    # else:
    #     raise ValueError("Maximum number of iterations reached")
    return x, _, r_k

if __name__ == "__main__":
    a = 0
    b = 3
    N = 100
    h = (b - a) / N
    x_i = np.linspace(a, b, N)

    exact = []
    for i in x_i:
        exact.append(utils.exactSolution(i))

    print(f'exact = {exact}')

    print(f'x_(-1) = {x_i[-1]}')
    print(f'x_0 = {x_i}')
    linear_solve = utils.solution(x_i, h)
    print(f'{linear_solve[0]}\n{linear_solve[1]}\n{linear_solve[2]}\n{linear_solve[3]}')

    xi_ = x_i.copy()
    res = solve(a=linear_solve[0], b=linear_solve[1], c=linear_solve[2], f=linear_solve[3], tol=1e-6, x=xi_)
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
