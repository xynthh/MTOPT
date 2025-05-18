import matplotlib.pyplot as plt
import numpy as np


def target_function(x):
    return x ** 3 + 3 * x ** 2 - 3


def fibonacci_number(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def required_fibonacci_n(length_ratio):
    a, b = 0, 1
    n = 1
    while b < length_ratio:
        a, b = b, a + b
        n += 1
    return n


def fibonacci_minimization():
    # Исходные данные
    interval_start = -1
    interval_end = 1
    epsilon = 0.2
    iteration = 0

    print("Метод Фибоначчи\n")
    print(f"Задан интервал: [{interval_start}, {interval_end}], точность ε = {epsilon}")

    initial_length = abs(interval_end - interval_start)
    N = required_fibonacci_n(initial_length / epsilon)
    print(f"Количество итераций N = {N}")

    point_y = interval_start + (fibonacci_number(N - 2) / fibonacci_number(N)) * (
            interval_end - interval_start
    )
    point_z = interval_start + (fibonacci_number(N - 1) / fibonacci_number(N)) * (
            interval_end - interval_start
    )

    print(f"\nНачальные точки:\n  y0 = {point_y}\n  z0 = {point_z}")

    intervals_history = [(interval_start, interval_end)]

    while iteration != (N - 3):
        f_y = target_function(point_y)
        f_z = target_function(point_z)
        print(f"\nИтерация {iteration}:")
        print(f"  f(y) = {f_y}, f(z) = {f_z}")

        if f_y <= f_z:
            interval_end = point_z
            point_z = point_y
            point_y = interval_start + (
                    fibonacci_number(N - iteration - 3)
                    / fibonacci_number(N - iteration - 1)
            ) * (interval_end - interval_start)
        else:
            interval_start = point_y
            point_y = point_z
            point_z = interval_start + (
                    fibonacci_number(N - iteration - 2)
                    / fibonacci_number(N - iteration - 1)
            ) * (interval_end - interval_start)

        iteration += 1
        intervals_history.append((interval_start, interval_end))
        print(f"  Новый интервал: [{interval_start}, {interval_end}]")

    final_y = (interval_start + interval_end) / 2
    final_z = final_y + epsilon

    f_final_y = target_function(final_y)
    f_final_z = target_function(final_z)
    print("\nЗавершающий шаг:")
    print(f"  y(N-1) = {final_y}, z(N-1) = {final_z}")
    print(f"  f(y) = {f_final_y}, f(z) = {f_final_z}")

    if f_final_y <= f_final_z:
        interval_end = final_z
    else:
        interval_start = final_y

    min_x = (interval_start + interval_end) / 2
    min_value = target_function(min_x)

    print(f"\nОтвет:\n  Минимум функции находится в x* ≈ {min_x}")
    print(f"  Значение функции: f(x*) = {min_value}")

    x_values = np.linspace(0.5, 3.5, 500)
    y_values = target_function(x_values)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="f(x)", linewidth=2)
    plt.axvline(x=min_x, color="red", linestyle="--", label=f"x* ≈ {min_x:.3f}")
    plt.scatter([min_x], [min_value], color="red", zorder=5)
    plt.title("Минимум функции методом Фибоначчи", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()

    for a, b in intervals_history:
        plt.axvspan(a, b, alpha=0.1, color="green")

    plt.show()


fibonacci_minimization()
