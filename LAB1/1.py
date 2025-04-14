import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3+3*x**2-3

a,b=-1,1
epsilon=0.2

def half_method (f,a,b,epsilon):
    iterations= []
    k=0

    while (b-a)>epsilon:
        l=abs(b-a)
        c=(a+b)/2
        y=a+l/4
        z=b-l/4
        fy,fc,fz=f(y),f(c),f(z)
        iterations.append((a,b,y,c,z,fy,fc,fz))
        if  fy<fc:
            b=c
            c=y
            print (f"Средняя точка: {c}", f"Модуль L: {l}",f"Значение функции в точке X_C_K: {fc}", f"Точка y: {y}", f"Значение f(y): {fy}", f"Точка z: {z}", f"Значение f(z): {fz}" )
        elif fz<fc:
            a=c
            c=z
            print (f"Средняя точка: {c}", f"Модуль L: {l}",f"Значение функции в точке X_C_K: {fc}", f"Точка y: {y}", f"Значение f(y): {fy}", f"Точка z: {z}", f"Значение f(z): {fz}" )
        else:
            a,b=y,z
            print (f"Средняя точка: {c}", f"Модуль L: {l}",f"Значение функции в точке X_C_K: {fc}", f"Точка y: {y}", f"Значение f(y): {fy}", f"Точка z: {z}", f"Значение f(z): {fz}" )
        k+=1
    x_min=(a+b)/2
    return x_min,f(x_min), k, iterations

x_min_half, f_min_half, k_half, steps_half= half_method(f, a, b, epsilon)

x_vals=np.linspace(a-0.5, b+0.5 ,500)
y_vals=f(x_vals)

plt.figure(figsize=(12,12))
plt.plot(x_vals, y_vals, label='f(x)')
for i, (a_i, b_i, y_i, c_i, z_i, fy_i, fc_i, fz_i) in enumerate(steps_half):
    plt.axvline(a_i,color='r',ls='--', alpha=0.3)
    plt.axvline(b_i,color='r',ls='--', alpha=0.3)
    plt.plot([y_i,c_i,z_i],[fy_i,fc_i,fz_i],'o',label=f"iter {i+1}" if i==0 else "")
plt.axvline(x_min_half,color='gray',ls='--',label=f"x*={x_min_half:.3f}")
plt.title("Метод половинного деления")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print (f"Минимум функции: {f_min_half}", f"Количество итераций: {k_half}", f"Точка минимума: {x_min_half}")
