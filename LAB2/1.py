import math
import numpy as np


# === Глобальные константы ===

SIZE_OF_BUFFER = 256


# === Вспомогательные функции ===


def f(a, b, c, x):
    """Вычисляет значение квадратичной функции двух переменных."""
    return a * x[0] ** 2 + b * x[0] * x[1] + c * x[1] ** 2


def grad(a, b, c, x):
    """Вычисляет градиент функции f по x и y."""
    return [2 * a * x[0] + b * x[1], 2 * c * x[1] + b * x[0]]  # df/dx  # df/dy


def norm(v):
    """Вычисляет евклидову норму вектора."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


# === Методы одномерной минимизации ===


def bisection_min(func, left, right, epsilon):
    """Метод половинного деления для одномерной минимизации."""
    while (right - left) > epsilon:
        mid = (left + right) / 2
        f_mid = func(mid)
        y = left + (right - left) / 4
        z = right - (right - left) / 4
        f_y = func(y)
        f_z = func(z)
        if f_y < f_mid:
            right = mid
        elif f_z < f_mid:
            left = mid
        else:
            left = y
            right = z
    return (left + right) / 2


def find_best_step_bisection(a, b, c, x, grad_x):
    """Поиск оптимального шага методом половинного деления."""
    epsilon = 0.2

    def func(t):
        x_new = [x[0] - t * grad_x[0], x[1] - t * grad_x[1]]
        return f(a, b, c, x_new)

    return bisection_min(func, 0, 100, epsilon)


# === Методы оптимизации ===


def gradient_descent_constant_step(a, b, c, eps_grad, eps_x):
    Message = "Метод градиентного спуска с постоянным шагом"
    x = [0.0, 0.5]  # начальная точка

    max_iter = 10
    init_t_k = 0.5
    t_k = 0
    result = x.copy()
    # флаг отвечающий за совпадение условий в шаге 9
    previuos_flag = False
    # флаг основного цикла
    flag = False

    print(
        f"\n1) Начальные данные:\nФункция - f(x,y) = {a}x^2 + {b}xy + {c}y^2. Найти минимум"
    )
    print(
        f"Начальная точка M({x[0]},{x[1]}), значение функции в начальной точке: f(M) = {f(a, b, c, x)}"
    )
    # шаг 2: положить k = 0
    k = 0

    print(f"2) Кладем флаг k = {k}")

    while not flag:
        # шаг 3: вычислить grad(f(x_k)))
        gradX = grad(a, b, c, x)
        print(
            f"3.{k}) Вычислим градиент в точке x_{k}:\ngrad(f(x_k) = ({gradX[0]}, {gradX[0]})"
        )

        grad_norm = norm(gradX)

        Message = f"4.{k}) Вычислим норму градианта функции в точке x_{k}: \n||grad(f(x_{k}))|| = {grad_norm} > {eps_grad:.2f} => переходим к шагу 5"
        print(Message)

        # Шаг 4: проверка условия окончания по градиенту
        if grad_norm < eps_grad:
            result = x.copy()
            flag = True

        Message = (
            f"5.{k}) Выполним проверку на переполнение итераций: k = {k} M = {max_iter}"
        )
        print(Message)
        # Шаг 5: проверка по итерациям
        if k >= max_iter:
            result = x.copy()
            flag = True

        Message = f"6.{k}) зададим начальный шаг t_k = {init_t_k:.1f}"
        print(Message)
        # шаг 6: начальное значение шага t_k = 0.5
        t_k = init_t_k

        # Шаг 7–8: найти x^{k+1}, удовлетворяющий условию убывания
        x_new = [0, 0]
        satisfying_conditions = False
        # количество итераций в цикле для шагов 7 и 8
        x_k = 0
        while not satisfying_conditions:
            x_new = [x[0] - t_k * gradX[0], x[1] - t_k * gradX[1]]
            f_new = f(a, b, c, x_new)
            f_curr = f(a, b, c, x)
            Message = f"7.{k}.{x_k}) вычислим точку x_k+1 = ({x_new[0]:.6f}, {x_new[1]:.6f}), f(x_k+1) = {f_new:.1f}"
            print(Message)
            Message = f"8.{k}.{x_k}) сравним полученные значения: f(x_k+1) = {f_new:.1f} и f(x_k) = {f_curr:.1f} (необходимо f(x_k+1) < f(x_k))"
            print(Message)
            if f_new - f_curr < 0:
                print("Условия выполнено, переходим к шагу 9")
                satisfying_conditions = True
            else:
                t_k /= 2.0

            x_k += 1

        # Шаг 9: проверка двойного условия окончания
        arg_diff = math.sqrt(pow(x_new[0] - x[0], 2) + pow(x_new[1] - x[1], 2))
        func_diff = abs(f(a, b, c, x_new) - f(a, b, c, x))

        if arg_diff < eps_x and func_diff < eps_x and previuos_flag:
            result = x_new.copy()
            flag = True
        elif arg_diff < eps_x and func_diff < eps_x:
            previuos_flag = True
        else:
            previuos_flag = False

        Message = f"9.{k}) Вычислим ||x_{k+1}-x_{k}|| = {arg_diff:.2f} (должно быть < {eps_x:.2f}) и |f(x_{k+1}) - f(x_{k})| = {func_diff:.2f} (должно быть < {eps_x:.2f})"
        print(Message + "\n")
        x = x_new.copy()
        k += 1

    suffix = " итерацию." if k == 1 else " итерации" if k in [2, 3] else " итераций. "
    print(f"Результат достигнут за {k}{suffix}")
    print(f"Точка минимума: x = {result[0]}, y = {result[1]}")
    return f(a, b, c, result)


def gradient_descent_steepest(a, b, c, eps_grad, eps_x):
    Message = "Метод наискорейшего градиентного спуска"
    x = [0.0, 0.5]  # начальная точка

    max_iter = 10
    init_t_k = 0
    t_k = 0
    result = x.copy()
    # флаг отвечающий за совпадение условий в шаге 9
    previuos_flag = False
    # флаг основного цикла
    flag = False

    print(
        f"\n1) Начальные данные:\nФункция - f(x,y) = {a}x^2 + {b}xy + {c}y^2. Найти минимум"
    )
    print(
        f"Начальная точка M({x[0]},{x[1]}), значение функции в начальной точке: f(M) = {f(a, b, c, x)}"
    )
    # шаг 2: положить k = 0
    k = 0

    print(f"2) Кладем флаг k = {k}")

    while not flag:
        # шаг 3: вычислить grad(f(x_k)))
        gradX = grad(a, b, c, x)
        print(
            f"3.{k}) Вычислим градиент в точке x_{k}:\ngrad(f(x_k) = ({gradX[0]}, {gradX[0]})"
        )

        grad_norm = norm(gradX)
        # заодно вычислим начальное значение t_k
        init_t_k = find_best_step_bisection(a, b, c, x, gradX)

        Message = f"4.{k}) Вычислим норму градианта функции в точке x_{k}: \n||grad(f(x_{k}))|| = {grad_norm} > {eps_grad:.2f} => переходим к шагу 5"
        print(Message)

        # Шаг 4: проверка условия окончания по градиенту
        if grad_norm < eps_grad:
            result = x.copy()
            flag = True

        Message = (
            f"5.{k}) Выполним проверку на переполнение итераций: k = {k} M = {max_iter}"
        )
        print(Message)
        # Шаг 5: проверка по итерациям
        if k >= max_iter:
            result = x.copy()
            flag = True

        Message = f"6.{k}) зададим начальный шаг t_k = {init_t_k:.1f}"
        print(Message)
        # шаг 6: начальное значение шага t_k
        t_k = init_t_k

        # Шаг 7–8: найти x^{k+1}, удовлетворяющий условию убывания
        x_new = [0, 0]
        satisfying_conditions = False
        # количество итераций в цикле для шагов 7 и 8
        x_k = 0
        while not satisfying_conditions:
            x_new = [x[0] - t_k * gradX[0], x[1] - t_k * gradX[1]]
            f_new = f(a, b, c, x_new)
            f_curr = f(a, b, c, x)
            Message = f"7.{k}.{x_k}) вычислим точку x_k+1 = ({x_new[0]:.6f}, {x_new[1]:.6f}), f(x_k+1) = {f_new:.1f}"
            print(Message)
            Message = f"8.{k}.{x_k}) сравним полученные значения: f(x_k+1) = {f_new:.1f} и f(x_k) = {f_curr:.1f} (необходимо f(x_k+1) < f(x_k))"
            print(Message)
            if f_new - f_curr < 0:
                print("Условия выполнено, переходим к шагу 9")
                satisfying_conditions = True
            else:
                t_k /= 2.0

            x_k += 1

        # Шаг 9: проверка двойного условия окончания
        arg_diff = math.sqrt(pow(x_new[0] - x[0], 2) + pow(x_new[1] - x[1], 2))
        func_diff = abs(f(a, b, c, x_new) - f(a, b, c, x))

        if arg_diff < eps_x and func_diff < eps_x and previuos_flag:
            result = x_new.copy()
            flag = True
        elif arg_diff < eps_x and func_diff < eps_x:
            previuos_flag = True
        else:
            previuos_flag = False

        Message = f"9.{k}) Вычислим ||x_{k+1}-x_{k}|| = {arg_diff:.2f} (должно быть < {eps_x:.2f}) и |f(x_{k+1}) - f(x_{k})| = {func_diff:.2f} (должно быть < {eps_x:.2f})"
        print(Message + "\n")
        x = x_new.copy()
        k += 1

    suffix = " итерацию." if k == 1 else " итерации" if k in [2, 3] else " итераций. "
    print(f"Результат достигнут за {k}{suffix}")
    print(f"Точка минимума: x = {result[0]}, y = {result[1]}")
    return f(a, b, c, result)


# === Точка входа ===


def main():
    eps_grad = 0.15
    eps_x = 0.2

    print("Задайте коэффициенты функции f(x, y) = ax^2 + bxy + cy^2")
    a = float(input("a: "))
    b = float(input("b: "))
    c = float(input("c: "))

    result = gradient_descent_constant_step(a, b, c, eps_grad, eps_x)
    print(f"Минимальное значение функции: {result}")

    return 0


if __name__ == "__main__":
    main()
