import utils
from math import *
import numpy as np
import plotly.graph_objects as go


def sumabs(x1, x2):
    return sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1))]))


def calc_r(A, x, b):
    n = len(x)
    Ax = [0] * n
    for i in range(n):
        for j in range(n):
            Ax[i] += A[i][j] * x[j]
    return [Ax[i] - b[i] for i in range(n)]


def prod(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


def yakobi(av, bv, cv, fv, tol=1e-6, x0=None, maxiter=None):
    n = len(fv)
    x0 = [0] * n if x0 is None else x0
    maxiter = 10 * n if maxiter is None else maxiter
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = bv[i]
        if i < n - 1:
            A[i + 1][i] = av[i + 1]
            A[i][i + 1] = cv[i]
    x1 = [-sum([A[i][j] / A[i][i] * x0[j] for j in range(0, n)]) + x0[i] + fv[i] / A[i][i] for i in range(0, n)]
    count = 0
    r_list = []
    r = calc_r(A, x0, fv)
    r_list.append(sqrt(prod(r, r)))
    while sumabs(x1, x0) > tol and count < maxiter:
        x0 = x1
        x1 = [-sum([A[i][j] / A[i][i] * x1[j] for j in range(0, n)]) + x1[i] + fv[i] / A[i][i] for i in range(0, n)]
        count += 1
        r = calc_r(A, x0, fv)
        r_list.append(sqrt(prod(r, r)))

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

    print(f'x_(-1) = {x_i[-1]}')
    linear_solve = utils.solution(x_i, h)
    print(f'{linear_solve[0]}\n{linear_solve[1]}\n{linear_solve[2]}\n{linear_solve[3]}')
    res = yakobi(cv=linear_solve[2], av=linear_solve[0], bv=linear_solve[1], fv=linear_solve[3])

    sol = res[0]
    error = []
    for i in range(len(sol)):
        error.append(abs(exact[i] - sol[i]))

    print("{:<20} | {:<25} | {:<20}".format("Exact Solution", "Approximate Solution", "Error"))
    print("-" * 75)

    for ex, ap, er in zip(exact, sol, error):
        print("{:<20} | {:<25} | {:<20}".format(ex, ap, er))

    print("\nMax Error: {}".format(max(error)))

    fig = go.Figure()

    # Добавляем точное решение
    fig.add_trace(go.Scatter(x=x_i, y=exact, mode='markers+lines', name='Точное решение', marker=dict(symbol='circle')))

    # Добавляем приближенное решение
    fig.add_trace(go.Scatter(x=x_i, y=sol, mode='markers+lines', name='Приближенное решение', marker=dict(symbol='x')))

    # Настройки макета
    fig.update_layout(title='График точного и приближенного решений',
                      xaxis=dict(title=f'Шаг {h}'),
                      yaxis=dict(title='Значение функции'),
                      legend=dict(x=0, y=1, traceorder='normal'))

    # Показываем график
    fig.show()


