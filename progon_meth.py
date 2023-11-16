import math

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import utils


def tridiagonal_matrix_algorithm(a, b, c, f):
    # Создаем вектора alpha и beta
    alpha = np.array([0, -c[0] / b[0]], dtype=np.float128, ndmin=1)
    beta = np.array([0, f[0] / b[0]], dtype=np.float128, ndmin=1)
    for i in range(1, len(b) - 1):
        delit = b[i] + a[i] * alpha[i]
        alpha = np.append(alpha, -c[i] / delit)
        beta = np.append(beta, (f[i] - a[i] * beta[i]) / delit)
    # alpha и beta заканчиваются n-ыми индексами

    # Ищем решенеи трехдиогоналной системы по методы прогонки уже
    x = np.array((a[-1] * beta[-1] - f[-1]) / (b[-1] - a[-1] * alpha[-1]), dtype=np.float128, ndmin=1)
    for i in range(len(b) - 1, 0, -1):
        x = np.append(np.array([(alpha[i] * x[0] + beta[i])], dtype=np.float128, ndmin=1),
                      x)

    return x


a = 0
b = 3
N = 500
h = (b - a) / N
x_i = np.linspace(a, b, N)
exact = []
for i in x_i:
    exact.append(utils.exactSolution(i))

print(f'x_(-1) = {x_i[-1]}')

linear_solve = utils.solution(x_i, h)
print(f'{linear_solve[0]}\n{linear_solve[1]}\n{linear_solve[2]}\n{linear_solve[3]}')
sol = tridiagonal_matrix_algorithm(linear_solve[0], linear_solve[1], linear_solve[2], linear_solve[3])

error = []
for i in range(len(sol)):
    error.append(abs(exact[i] - sol[i]))


plt.figure(figsize=(9, 6.75))
plt.plot(x_i, exact, label='Точное решение')
plt.plot(x_i, sol, label='Приближенное решение')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

print("{:<20} | {:<25} | {:<20}".format("Exact Solution", "Approximate Solution", "Error"))
print("-" * 75)

for ex, ap, er in zip(exact, sol, error):
    print("{:<20} | {:<25} | {:<20}".format(ex, ap, er))

print("\nMax Error: {}".format(max(error)))
