import numpy as np
from scipy.optimize import minimize


# =============================================================================
# ВАРИАНТ 15
# Целевая функция: f(x) = 6x₁² + 2x₂² - 17 → min
# Ограничение: g₁(x) = 8x₁ + x₂ - 7 = 0
# Параметры: ε = 0,05, r⁰ = 0,5, C = 10
# =============================================================================


def f(x):
    """Целевая функция: f(x) = 6x₁² + 2x₂² - 17"""
    return 6 * x[0] ** 2 + 2 * x[1] ** 2 - 17


def g1(x):
    """Ограничение: g₁(x) = 8x₁ + x₂ - 7"""
    return 8 * x[0] + x[1] - 7


def grad_f(x):
    """Градиент целевой функции"""
    return np.array([12 * x[0], 4 * x[1]])


def grad_g1():
    """Градиент ограничения"""
    return np.array([8, 1])


def penalty_gradient(x, r):
    """Градиент вспомогательной функции штрафа"""
    return grad_f(x) + r * g1(x) * grad_g1()


# =============================================================================
# МЕТОД ШТРАФОВ
# =============================================================================


def penalty_auxiliary_function(x, r):
    """
    Вспомогательная функция для метода штрафов:
    F(x,r) = f(x) + r/2 * [g₁(x)]²
    """
    return f(x) + r / 2 * g1(x) ** 2


def penalty_method(x0=None, r0=0.5, C=10, eps=0.05, max_iter=20):
    """
    Метод штрафов для решения задачи условной оптимизации

    Параметры:
    x0 - начальная точка (если None, то выбирается автоматически)
    r0 - начальное значение параметра штрафа
    C - коэффициент увеличения параметра штрафа
    eps - точность для условия остановки
    max_iter - максимальное количество итераций

    Возвращает:
    result - найденная точка минимума
    f_val - значение целевой функции в найденной точке
    iterations - количество итераций
    history - история точек и значений функций
    """
    print("=== МЕТОД ШТРАФОВ ===")
    print(f"Начальные параметры: r⁰ = {r0}, C = {C}, ε = {eps}")

    # Начальная точка (вне множества допустимых решений)
    if x0 is None:
        x0 = np.array([0.0, 1.0])

    print(f"Начальная точка: x⁰ = {x0}")
    print(f"f(x⁰) = {f(x0):.6f}, g₁(x⁰) = {g1(x0):.6f}")
    print()

    x_k = x0.copy()
    r_k = r0
    history = {
        "points": [x_k.copy()],
        "f_values": [f(x_k)],
        "g_values": [g1(x_k)],
        "penalty_values": [0],
        "r_values": [r_k],
    }

    for k in range(max_iter):
        print(f"--- Итерация {k} ---")
        print(f"r^{k} = {r_k}")

        # Составляем вспомогательную функцию
        def F_penalty(x):
            return penalty_auxiliary_function(x, r_k)

        def current_penalty_gradient(x):
            return penalty_gradient(x, r_k)

        # Минимизируем вспомогательную функцию
        result = minimize(
            F_penalty, x_k, method="Newton-CG", jac=current_penalty_gradient
        )
        x_new = result.x

        print(f"x^*({r_k}) = ({x_new[0]:.6f}, {x_new[1]:.6f})")
        print(f"f(x^*({r_k})) = {f(x_new):.6f}")
        print(f"g₁(x^*({r_k})) = {g1(x_new):.6f}")

        # Вычисляем штрафную функцию
        P_value = r_k / 2 * g1(x_new) ** 2
        print(f"P(x^*({r_k}), r^{k}) = {P_value:.6f}")

        # Проверяем условие окончания
        if P_value <= eps:
            print(f"Условие окончания выполнено: P = {P_value:.6f} ≤ {eps}")
            history["points"].append(x_new.copy())
            history["f_values"].append(f(x_new))
            history["g_values"].append(g1(x_new))
            history["penalty_values"].append(P_value)
            history["r_values"].append(r_k)
            print(f"Результат: x* = ({x_new[0]:.6f}, {x_new[1]:.6f})")
            print(f"f(x*) = {f(x_new):.6f}")
            print(f"Количество итераций: {k + 1}")
            return x_new, f(x_new), k + 1, history

        # Увеличиваем параметр штрафа и переходим к следующей итерации
        r_k *= C
        x_k = x_new.copy()

        history["points"].append(x_k.copy())
        history["f_values"].append(f(x_k))
        history["g_values"].append(g1(x_k))
        history["penalty_values"].append(P_value)
        history["r_values"].append(r_k)
        print()

    print(f"Достигнуто максимальное количество итераций: {max_iter}")
    return x_k, f(x_k), max_iter, history


# =============================================================================
# МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ
# =============================================================================


def barrier_auxiliary_function_inverse(x, r):
    """
    Вспомогательная функция для метода барьерных функций (обратная):
    F(x,r) = f(x) - r * 1/g₁(x)
    Работает только для ограничений типа g(x) ≤ 0
    """
    g_val = -g1(x)  # Преобразуем g₁(x) = 0 в g₁(x) ≤ 0
    if g_val <= 0:
        return np.inf  # Барьер: не допускаем выход за границу
    return f(x) - r / g_val


def barrier_auxiliary_function_log(x, r):
    """
    Вспомогательная функция для метода барьерных функций (логарифмическая):
    F(x,r) = f(x) - r * ln(-g₁(x))
    Работает только для ограничений типа g(x) ≤ 0
    """
    g_val = -g1(x)  # Преобразуем g₁(x) = 0 в g₁(x) ≤ 0
    if g_val <= 0:
        return np.inf  # Барьер: не допускаем выход за границу
    return f(x) - r * np.log(g_val)


def barrier_method(barrier_type="inverse", x0=None, r0=1.0, C=5, eps=0.05, max_iter=20):
    """
    Метод барьерных функций для решения задачи условной оптимизации

    Параметры:
    barrier_type - тип барьерной функции ('inverse' или 'log')
    x0 - начальная точка внутри допустимой области
    r0 - начальное значение параметра штрафа
    C - коэффициент уменьшения параметра штрафа
    eps - точность для условия остановки
    max_iter - максимальное количество итераций

    Возвращает:
    result - найденная точка минимума
    f_val - значение целевой функции в найденной точке
    iterations - количество итераций
    history - история точек и значений функций
    """
    print(f"=== МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ ({barrier_type.upper()}) ===")
    print(f"Начальные параметры: r⁰ = {r0}, C = {C}, ε = {eps}")

    # Выбираем функцию барьера
    if barrier_type == "inverse":
        F_barrier = barrier_auxiliary_function_inverse
    else:
        F_barrier = barrier_auxiliary_function_log

    # Начальная точка ВНУТРИ допустимой области
    # Для ограничения g₁(x) = 8x₁ + x₂ - 7 = 0, нужно чтобы -g₁(x) > 0
    # То есть 8x₁ + x₂ - 7 < 0, или 8x₁ + x₂ < 7
    if x0 is None:
        x0 = np.array([0.5, 0.5])  # Проверим: 8*0.5 + 0.5 = 4.5 < 7 ✓

    # Проверяем, что начальная точка внутри области
    if -g1(x0) <= 0:
        print("ОШИБКА: Начальная точка не находится внутри допустимой области!")
        print(f"g₁(x⁰) = {g1(x0)}, требуется g₁(x⁰) < 0")
        return None, None, 0, None

    print(f"Начальная точка: x⁰ = {x0}")
    print(f"f(x⁰) = {f(x0):.6f}, g₁(x⁰) = {g1(x0):.6f}")
    print()

    x_k = x0.copy()
    r_k = r0
    history = {
        "points": [x_k.copy()],
        "f_values": [f(x_k)],
        "g_values": [g1(x_k)],
        "barrier_values": [0],
        "r_values": [r_k],
    }

    for k in range(max_iter):
        print(f"--- Итерация {k} ---")
        print(f"r^{k} = {r_k}")

        # Составляем вспомогательную функцию
        def F_barrier_k(x):
            return F_barrier(x, r_k)

        # Минимизируем вспомогательную функцию
        # Используем ограничения, чтобы не выйти за границу области
        def constraint(x):
            return -g1(x)  # должно быть > 0

        cons = {"type": "ineq", "fun": constraint}
        result = minimize(F_barrier_k, x_k, method="SLSQP", constraints=cons)

        if not result.success:
            print("Не удалось найти минимум на данной итерации")
            break

        x_new = result.x

        print(f"x^*({r_k}) = ({x_new[0]:.6f}, {x_new[1]:.6f})")
        print(f"f(x^*({r_k})) = {f(x_new):.6f}")
        print(f"g₁(x^*({r_k})) = {g1(x_new):.6f}")

        # Вычисляем барьерную функцию
        if barrier_type == "inverse":
            P_value = abs(r_k / (-g1(x_new)))
        else:
            P_value = abs(r_k * np.log(-g1(x_new)))

        print(f"P(x^*({r_k}), r^{k}) = {P_value:.6f}")

        # Проверяем условие окончания
        if P_value <= eps:
            print(f"Условие окончания выполнено: P = {P_value:.6f} ≤ {eps}")
            history["points"].append(x_new.copy())
            history["f_values"].append(f(x_new))
            history["g_values"].append(g1(x_new))
            history["barrier_values"].append(P_value)
            history["r_values"].append(r_k)
            print(f"Результат: x* = ({x_new[0]:.6f}, {x_new[1]:.6f})")
            print(f"f(x*) = {f(x_new):.6f}")
            print(f"Количество итераций: {k + 1}")
            return x_new, f(x_new), k + 1, history

        # Уменьшаем параметр штрафа и переходим к следующей итерации
        r_k /= C
        x_k = x_new.copy()

        history["points"].append(x_k.copy())
        history["f_values"].append(f(x_k))
        history["g_values"].append(g1(x_k))
        history["barrier_values"].append(P_value)
        history["r_values"].append(r_k)
        print()

    print(f"Достигнуто максимальное количество итераций: {max_iter}")
    return x_k, f(x_k), max_iter, history


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================


def main():
    """Главная функция программы"""
    print("ЛАБОРАТОРНАЯ РАБОТА №4: МЕТОДЫ УСЛОВНОЙ ОПТИМИЗАЦИИ")
    print("ВАРИАНТ 15")
    print("Целевая функция: f(x) = 6x₁² + 2x₂² - 17 → min")
    print("Ограничение: g₁(x) = 8x₁ + x₂ - 7 = 0")
    print("Параметры: ε = 0,05, r⁰ = 0,5, C = 10")
    print("\n" + "=" * 60)

    # 1. Метод штрафов
    print("\n1. РЕШЕНИЕ МЕТОДОМ ШТРАФОВ")
    x_penalty, f_penalty, iter_penalty, history_penalty = penalty_method()

    # 2. Метод барьерных функций (обратная функция)
    print("\n2. РЕШЕНИЕ МЕТОДОМ БАРЬЕРНЫХ ФУНКЦИЙ (ОБРАТНАЯ ФУНКЦИЯ)")
    x_barrier_inv, f_barrier_inv, iter_barrier_inv, history_barrier_inv = (
        barrier_method("inverse")
    )

    # 3. Метод барьерных функций (логарифмическая функция)
    print("\n3. РЕШЕНИЕ МЕТОДОМ БАРЬЕРНЫХ ФУНКЦИЙ (ЛОГАРИФМИЧЕСКАЯ ФУНКЦИЯ)")
    x_barrier_log, f_barrier_log, iter_barrier_log, history_barrier_log = (
        barrier_method("log")
    )

    # 4. Сравнение результатов
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    print(f"Метод штрафов:           x* = ({x_penalty[0]:.6f}, {x_penalty[1]:.6f})")
    print(f"                         f* = {f_penalty:.6f}, итераций: {iter_penalty}")
    print(f"                         g₁(x*) = {g1(x_penalty):.6f}")
    print()
    if x_barrier_inv is not None:
        print(
            f"Барьеры (обратная):      x* = ({x_barrier_inv[0]:.6f}, {x_barrier_inv[1]:.6f})"
        )
        print(
            f"                         f* = {f_barrier_inv:.6f}, итераций: {iter_barrier_inv}"
        )
        print(f"                         g₁(x*) = {g1(x_barrier_inv):.6f}")
        print()
    if x_barrier_log is not None:
        print(
            f"Барьеры (логарифм):      x* = ({x_barrier_log[0]:.6f}, {x_barrier_log[1]:.6f})"
        )
        print(
            f"                         f* = {f_barrier_log:.6f}, итераций: {iter_barrier_log}"
        )
        print(f"                         g₁(x*) = {g1(x_barrier_log):.6f}")
        print()


if __name__ == "__main__":
    main()
