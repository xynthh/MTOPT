import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# === Целевая функция и её производные ===


def f(x, a, b, c):
    """Вычисляет значение квадратичной функции двух переменных: a*x[0]^2 + b*x[0]*x[1] + c*x[1]^2."""
    return a * x[0] ** 2 + b * x[0] * x[1] + c * x[1] ** 2


def grad_f(x, a, b, c):
    """Вычисляет градиент функции: (2a*x₁ + b*x₂, b*x₁ + 2c*x₂)."""
    return np.array([2 * a * x[0] + b * x[1], b * x[0] + 2 * c * x[1]])


def hessian(x, a, b, c):
    """Вычисляет матрицу Гессе: [[2a, b], [b, 2c]]."""
    return np.array([[2 * a, b], [b, 2 * c]])


# === Методы оптимизации ===


def newton_method(f, grad_f, hessian, a, b, c, x0, eps1, eps2, max_iter):
    """
    Реализация метода Ньютона для минимизации функции.

    Параметры:
    f - целевая функция
    grad_f - функция для вычисления градиента
    hessian - функция для вычисления матрицы Гессе
    a, b, c - коэффициенты функции
    x0 - начальная точка
    eps1 - точность для нормы градиента
    eps2 - точность для изменения аргумента и значения функции
    max_iter - максимальное количество итераций

    Возвращает:
    (точка минимума, значение минимума, количество итераций, история точек, история значений)
    """
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x, a, b, c)]

    print("Начинаем метод Ньютона")
    print(f"Начальная точка: x⁰ = {x}, f(x⁰) = {f(x, a, b, c)}")
    print(f"Параметры: ε₁ = {eps1}, ε₂ = {eps2}, M = {max_iter}\n")

    for k in range(max_iter):
        # Шаг 3: Вычислить градиент
        grad = grad_f(x, a, b, c)
        grad_norm = np.linalg.norm(grad)
        print(f"Итерация {k}:")
        print(f"Градиент: ∇f(x{k}) = {grad}")
        print(f"Норма градиента: ||∇f(x{k})|| = {grad_norm:.6f}")

        # Шаг 4: Проверить критерий окончания
        if grad_norm <= eps1:
            print(
                f"Критерий окончания выполнен: ||∇f(x{k})|| = {grad_norm:.6f} ≤ {eps1}"
            )
            return x, f(x, a, b, c), k, x_history, f_history

        # Шаг 5: Проверить число итераций
        if k == max_iter - 1:
            print(f"Достигнуто максимальное число итераций: {max_iter}")
            return x, f(x, a, b, c), k + 1, x_history, f_history

        # Шаг 6: Вычислить матрицу Гессе
        H = hessian(x, a, b, c)
        print(f"Матрица Гессе H(x{k}) = \n{H}")

        # Шаг 7: Вычислить обратную матрицу Гессе
        try:
            H_inv = np.linalg.inv(H)
            print(f"Обратная матрица Гессе H⁻¹(x{k}) = \n{H_inv}")
        except np.linalg.LinAlgError:
            print("Матрица Гессе вырождена или плохо обусловлена")
            d = -grad
            print(f"Используем направление антиградиента: d{k} = -∇f(x{k}) = {d}")
            t_k = 1.0
            while True:
                x_new = x + t_k * d
                if f(x_new, a, b, c) < f(x, a, b, c):
                    break
                t_k /= 2.0
            print(f"Найден шаг t{k} = {t_k}")
            print(f"Новая точка: x{k + 1} = {x_new}, f(x{k + 1}) = {f(x_new, a, b, c):.6f}")

            dx = np.linalg.norm(x_new - x)
            df = abs(f(x_new, a, b, c) - f(x, a, b, c))
            print(f"||x{k + 1} - x{k}|| = {dx:.6f}, |f(x{k + 1}) - f(x{k})| = {df:.6f}")

            x = x_new
            x_history.append(x.copy())
            f_history.append(f(x, a, b, c))
            continue

        # Шаг 8-9: Проверить положительную определенность и определить направление
        eigenvalues = np.linalg.eigvals(H)
        if np.all(eigenvalues > 0):
            print(
                f"Матрица Гессе положительно определена (собственные значения: {eigenvalues})"
            )
            # Метод Ньютона
            d = -H_inv @ grad
            print(f"Направление Ньютона: d{k} = -H⁻¹∇f(x{k}) = {d}")
            t_k = 1.0
        else:
            print(
                f"Матрица Гессе не положительно определена (собственные значения: {eigenvalues})"
            )
            # Метод градиентного спуска
            d = -grad
            print(f"Используем направление антиградиента: d{k} = -∇f(x{k}) = {d}")
            t_k = 1.0
            while True:
                x_new = x + t_k * d
                if f(x_new, a, b, c) < f(x, a, b, c):
                    break
                t_k /= 2.0

        # Шаг 10: Найти новую точку
        x_new = x + t_k * d
        if t_k != 1.0:
            print(f"Подобран шаг t{k} = {t_k}")
        print(f"Новая точка: x{k + 1} = {x_new}, f(x{k + 1}) = {f(x_new, a, b, c):.6f}")

        # Шаг 11: Проверить условия останова
        dx = np.linalg.norm(x_new - x)
        df = abs(f(x_new, a, b, c) - f(x, a, b, c))
        print(f"||x{k + 1} - x{k}|| = {dx:.6f}, |f(x{k + 1}) - f(x{k})| = {df:.6f}")

        if dx < eps2 and df < eps2:
            if k > 0:  # Проверка для k и k-1
                prev_dx = np.linalg.norm(x - x_history[-2])
                prev_df = abs(f(x, a, b, c) - f_history[-2])
                if prev_dx < eps2 and prev_df < eps2:
                    print(f"Критерии окончания выполнены дважды последовательно")
                    return (
                        x_new,
                        f(x_new, a, b, c),
                        k + 1,
                        x_history + [x_new],
                        f_history + [f(x_new, a, b, c)],
                    )

        x = x_new
        x_history.append(x.copy())
        f_history.append(f(x, a, b, c))
        print()  # Пустая строка для разделения итераций

    return x, f(x, a, b, c), max_iter, x_history, f_history


# === Визуализация ===


def plot_optimization_results(a, b, c, x0, x_min, f_min, x_history):
    """
    Построение графиков оптимизационного процесса.

    Параметры:
    a, b, c - коэффициенты целевой функции
    x0 - начальная точка
    x_min - найденная точка минимума
    f_min - значение минимума
    x_history - история точек итерационного процесса
    """
    fig = plt.figure(figsize=(12, 5))

    # 3D поверхность
    ax1 = fig.add_subplot(121, projection="3d")
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]), a, b, c)

    surf = ax1.plot_surface(
        X1, X2, Z, cmap=cm.coolwarm, alpha=0.8, rstride=1, cstride=1
    )
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_zlabel("f(x)")
    ax1.set_title(f"Функция f(x) = {a}x₁² + {b}x₁x₂ + {c}x₂²")

    # Отметить точки итераций
    x_hist = np.array(x_history)
    for i, x in enumerate(x_hist):
        ax1.scatter(x[0], x[1], f(x, a, b, c), color="red", s=50)

    # Отметить минимум
    ax1.scatter(x_min[0], x_min[1], f_min, color="green", s=100, label="Минимум")
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])

    # Автоматический выбор предела оси Z
    max_z = np.max(Z)
    ax1.set_zlim([0, min(max_z, 10)])

    # Контурный график
    ax2 = fig.add_subplot(122)
    levels = np.linspace(0, min(max_z, 5), 20)
    contour = ax2.contour(X1, X2, Z, levels=levels, cmap=cm.coolwarm)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Отметить путь итераций
    for i in range(len(x_hist) - 1):
        ax2.plot(
            [x_hist[i][0], x_hist[i + 1][0]],
            [x_hist[i][1], x_hist[i + 1][1]],
            "r-o",
            linewidth=1.5,
        )

    # Отметить начальную точку и минимум
    ax2.scatter(x0[0], x0[1], color="blue", s=100, marker="o", label="Начальная точка")
    ax2.scatter(x_min[0], x_min[1], color="green", s=100, marker="*", label="Минимум")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_title("Линии уровня и путь итераций")
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim([-2, 2])
    ax2.set_ylim([-2, 2])

    plt.tight_layout()
    plt.savefig("newton_method_optimization.png")  # Сохранить изображение
    plt.show()


# === Точка входа ===


def main():
    """Основная функция программы."""
    # Запрос пользовательских параметров
    print("Введите коэффициенты для функции f(x, y) = a*x^2 + b*x*y + c*y^2")
    a = float(input("a: "))
    b = float(input("b: "))
    c = float(input("c: "))

    # Начальные параметры
    x0 = np.array([1.5, 0.5])
    eps1 = 0.15
    eps2 = 0.2
    max_iter = 10

    # Поиск минимума
    x_min, f_min, iterations, x_history, f_history = newton_method(
        f, grad_f, hessian, a, b, c, x0, eps1, eps2, max_iter
    )

    print("\nРезультат:")
    print(f"Точка минимума: x* = {x_min}")
    print(f"Минимальное значение функции: f(x*) = {f_min:.6f}")
    print(f"Количество итераций: {iterations}")

    # Визуализация результатов
    plot_optimization_results(a, b, c, x0, x_min, f_min, x_history)

    return 0


if __name__ == "__main__":
    main()
