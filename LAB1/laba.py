import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 3 + 3 * x ** 2 - 3


a, b = -1, 1
epsilon = 0.2


# Метод 1: Метод половинного деления
def number_1():
    print("\n--- Метод половинного деления ---\n")

    # Метод половинного деления
    def half_method(f, a, b, epsilon):
        iterations = []
        k = 0

        while (b - a) > epsilon:
            l = abs(b - a)
            c = (a + b) / 2
            y = a + l / 4
            z = b - l / 4
            fy, fc, fz = f(y), f(c), f(z)
            iterations.append((a, b, y, c, z, fy, fc, fz))
            if fy < fc:
                b = c
                c = y
                print(
                    f"Средняя точка: {c}",
                    f"Модуль L: {l}",
                    f"Значение функции в точке X_C_K: {fc}",
                    f"Точка y: {y}",
                    f"Значение f(y): {fy}",
                    f"Точка z: {z}",
                    f"Значение f(z): {fz}",
                )
            elif fz < fc:
                a = c
                c = z
                print(
                    f"Средняя точка: {c}",
                    f"Модуль L: {l}",
                    f"Значение функции в точке X_C_K: {fc}",
                    f"Точка y: {y}",
                    f"Значение f(y): {fy}",
                    f"Точка z: {z}",
                    f"Значение f(z): {fz}",
                )
            else:
                a, b = y, z
                print(
                    f"Средняя точка: {c}",
                    f"Модуль L: {l}",
                    f"Значение функции в точке X_C_K: {fc}",
                    f"Точка y: {y}",
                    f"Значение f(y): {fy}",
                    f"Точка z: {z}",
                    f"Значение f(z): {fz}",
                )
            k += 1
        x_min = (a + b) / 2
        return x_min, f(x_min), k, iterations

    x_min_half, f_min_half, k_half, steps_half = half_method(f, a, b, epsilon)

    x_vals = np.linspace(a - 0.5, b + 0.5, 500)
    y_vals = f(x_vals)

    plt.figure(figsize=(12, 12))
    plt.plot(x_vals, y_vals, label="f(x)")
    for i, (a_i, b_i, y_i, c_i, z_i, fy_i, fc_i, fz_i) in enumerate(steps_half):
        plt.axvline(a_i, color="r", ls="--", alpha=0.3)
        plt.axvline(b_i, color="r", ls="--", alpha=0.3)
        plt.plot(
            [y_i, c_i, z_i],
            [fy_i, fc_i, fz_i],
            "o",
            label=f"iter {i + 1}" if i == 0 else "",
        )
    plt.axvline(x_min_half, color="gray", ls="--", label=f"x*={x_min_half:.3f}")
    plt.title("Метод половинного деления")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(
        f"Минимум функции: {f_min_half}",
        f"Количество итераций: {k_half}",
        f"Точка минимума: {x_min_half}",
    )


# Метод 2: Метод золотого сечения
def number_2():
    print("\n--- Метод золотого сечения ---\n")

    # Метод золотого сечения с визуализацией
    def golden_section_search_with_plot(f, a0, b0, epsilon):
        golden_ratio = (3 - np.sqrt(5)) / 2  # ≈ 0.382 - константа из методички
        a, b = a0, b0
        k = 0
        history = []

        y = a + golden_ratio * (b - a)
        z = a + b - y
        fy = f(y)
        fz = f(z)

        while abs(b - a) > epsilon:
            history.append((a, b, y, z, fy, fz))
            if fy <= fz:
                b = z
                z = y
                fz = fy
                y = a + golden_ratio * (b - a)
                fy = f(y)
                print(
                    f"Точка y: {y}",
                    f"Значение f(y): {fy}",
                    f"Точка z: {z}",
                    f"Значение f(z): {fz}",
                    k,
                )
            else:
                a = y
                y = z
                fy = fz
                z = a + b - y
                fz = f(z)
                print(
                    f"Точка y: {y}",
                    f"Значение f(y): {fy}",
                    f"Точка z: {z}",
                    f"Значение f(z): {fz}",
                    k,
                )
            k += 1

        x_star = (a + b) / 2
        f_star = f(x_star)

        # График
        x_vals = np.linspace(a0 - 0.5, b0 + 0.5, 500)
        y_vals = f(x_vals)
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label="f(x)")

        for i, (a_i, b_i, y_i, z_i, _, _) in enumerate(history):
            plt.axvline(a_i, color="gray", linestyle="--", alpha=0.3)
            plt.axvline(b_i, color="gray", linestyle="--", alpha=0.3)
            plt.plot(
                [y_i, z_i],
                [f(y_i), f(z_i)],
                "o",
                label=f"Итерация {i + 1}" if i == 0 else "",
            )

        plt.axvline(x_star, color="red", linestyle="--", label=f"x* ≈ {x_star:.3f}")
        plt.title("Метод золотого сечения")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return x_star, f_star, k

    x_star, f_star, k = golden_section_search_with_plot(f, a, b, epsilon)
    print(f"Минимум функции: {f_star}")
    print(f"Количество итераций: {k}")
    print(f"Точка минимума: {x_star}")


# Метод 3: Метод Фибоначчи
def number_3():
    print("\n--- Метод Фибоначчи ---\n")

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

    # Метод Фибоначчи с визуализацией
    def fibonacci_minimization():
        # Исходные данные
        interval_start = -1
        interval_end = 1
        epsilon = 0.1
        iteration = 0

        print("Метод Фибоначчи\n")
        print(
            f"Задан интервал: [{interval_start}, {interval_end}], точность ε = {epsilon}"
        )

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
            f_y = f(point_y)
            f_z = f(point_z)
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

        f_final_y = f(final_y)
        f_final_z = f(final_z)
        print("\nЗавершающий шаг:")
        print(f"  y(N-1) = {final_y}, z(N-1) = {final_z}")
        print(f"  f(y) = {f_final_y}, f(z) = {f_final_z}")

        if f_final_y <= f_final_z:
            interval_end = final_z
        else:
            interval_start = final_y

        min_x = (interval_start + interval_end) / 2
        min_value = f(min_x)

        print(f"\nОтвет:\n  Минимум функции находится в x* ≈ {min_x}")
        print(f"  Значение функции: f(x*) = {min_value}")

        x_values = np.linspace(0.5, 3.5, 500)
        y_values = f(x_values)

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


def main():
    print("Лабораторная работа №1 - Одномерная минимизация")
    print("Функция f(x) = x^3 + 3x^2 - 3")
    print("Интервал [a, b] = [-1, 1], точность ε = 0.2")
    print("\nВыберите метод решения:")
    print("1. Метод половинного деления")
    print("2. Метод золотого сечения")
    print("3. Метод Фибоначчи")

    choice = input("\nВведите номер метода (1-3): ")

    if choice == "1":
        number_1()
    elif choice == "2":
        number_2()
    elif choice == "3":
        number_3()
    else:
        print("Некорректный выбор. Пожалуйста, выберите число от 1 до 3.")


if __name__ == "__main__":
    main()
