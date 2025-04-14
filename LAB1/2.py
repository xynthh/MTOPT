import numpy as np
import matplotlib.pyplot as plt


# Функция из задания
def f(x):
    return x**3+3*x**2-3


# Интервал и точность
a0, b0 = -1, 1
epsilon = 0.2


# Метод золотого сечения с визуализацией
def golden_section_search_with_plot(f, a0, b0, epsilon):
    phi = (3-np.sqrt(5))/2
    a, b = a0, b0
    k = 0
    history = []


    y = a + (1 - phi) * (b - a)
    z = a + phi * (b - a)
    fy = f(y)
    fz = f(z)


    while abs(b - a) > epsilon:
        history.append((a, b, y, z, fy, fz))
        if fy <= fz:
            b = z
            z = y
            fz = fy
            y = a + (1 - phi) * (b - a)
            fy = f(y)
            print( f"Точка y: {y}",
                  f"Значение f(y): {fy}", f"Точка z: {z}", f"Значение f(z): {fz}", k)
        else:
            a = y
            y = z
            fy = fz
            z = a + phi * (b - a)
            fz = f(z)
            print(f"Точка y: {y}",
                  f"Значение f(y): {fy}", f"Точка z: {z}", f"Значение f(z): {fz}", k)
        k += 1


    x_star = (a + b) / 2
    f_star = f(x_star)


    # График
    x_vals = np.linspace(a0 - 0.5, b0 + 0.5, 500)
    y_vals = f(x_vals)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)')


    for i, (a_i, b_i, y_i, z_i, _, _) in enumerate(history):
        plt.axvline(a_i, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(b_i, color='gray', linestyle='--', alpha=0.3)
        plt.plot([y_i, z_i], [f(y_i), f(z_i)], 'o', label=f"Итерация {i+1}" if i == 0 else "")


    plt.axvline(x_star, color='red', linestyle='--', label=f"x* ≈ {x_star:.3f}")
    plt.title("Метод золотого сечения")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return x_star, f_star, k


x_star, f_star, k = golden_section_search_with_plot(f, a0, b0, epsilon)
print(f"Минимум функции: {f_star}")
print(f"Количество итераций: {k}")
print(f"Точка минимума: {x_star}")