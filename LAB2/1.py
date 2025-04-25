import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sympy as sp

# Шаг 1

# Параметры
x0 = np.array([0, 0.5])  # Начальная точка
eps1 = 0.15  # Точность для нормы градиента
eps2 = 0.2  # Условие остановки
M = 10  # Число итераций
step_size = 0.5


def f(x):
    return x[0] ** 2 + 0.6 * x[0] * x[1] + 6 * x[1] ** 2


def grad_f(x):
    return np.array([2 * x[0] + 0.6 * x[1], 0.6 * x[0] + 12 * x[1]])


def gradient_descent_with_constant_step(f, grad_f, x0, eps1, eps2, M, step_size):
    x = x0
    iterations = []
    iterations.append((x.copy(), f(x)))
    # Шаг 2
    for i in range(M):
        # Шаг 3 | вычисление градиента
        grad = grad_f(x)

        # Шаг 4
        grad_norm = np.linalg.norm(grad)
        if grad_norm < eps1:
            print(
                f"Критерий окончания по норме градиента выполнен: ||∇f|| = {grad_norm:.6f} < {eps1}"
            )
            return x, f(x), i + 1, iterations
        # Шаг 5
        if i == M - 1:
            print(
                f"Критерий окончания по числу итераций выполнен: количество итераций = {M}"
            )
            return x, f(x), i + 1, iterations
        # ....


x_min, f_min, iterations_count, history = gradient_descent_with_constant_step(
    f, grad_f, x0, eps1, eps2, M, step_size
)

print("\nРезультат:")
print(f"Точка минимума: x* ≈ {x_min}")
print(f"Значение функции: f(x*) ≈ {f_min:.6f}")
print(f"Количество итераций: {iterations_count}")
