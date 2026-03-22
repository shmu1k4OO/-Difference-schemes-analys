import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(x, t):
    return -2 


def phi(x):
    return x**2 + 1


def dphi(x):
    return 2 * np.ones_like(x)


def g_0(t):
    return 2 * t + 1


def g_L(t, L):
    return L**2 + 2 * t + 1


def exact_solve(x, t):
    return x ** 2 + 2 * t + 1

# X
L = 10

nx = 100

dx = L / (nx - 1)

x = np.linspace(0, L, nx)

# T
T = 100

nt = 1000

dt = T / (nt - 1)

t = np.linspace(0, T, nt)

u = np.zeros((nt, nx))

v = np.array(dphi(x))

alpha = 1

print(alpha * dt**2 / dx**2 < 1)

def solve_with_method_1(u, x, t, v, dt, dx, nx, alpha, f, L):

    u[0, :] = phi(x)
    u[:, 0] = g_0(t)
    u[:, -1] = g_L(t, L)

    u_xx = np.zeros(nx)
    for i in range(1, nx-1):
        u_xx[i] = (u[0, i-1] - 2*u[0, i] + u[0, i+1]) / dx**2

    u[1, :] = u[0, :] + dt * v + (dt**2/2) * (alpha * u_xx + f(x, 0))

    u[1, 0] = g_0(t[1])
    u[1, -1] = g_L(t[1], L)

    for j in range(1, nt-1):
        for i in range(1, nx-1):
            u[j+1, i] = 2 * u[j, i] - u[j-1, i] + dt**2 / dx**2 * alpha * (u[j, i+1] - 2*u[j, i] + u[j, i-1]) + dt**2 *f(x[i], t[j])
    
    return u

X, T_grid = np.meshgrid(x, t)
u_exact = exact_solve(X, T_grid)

error_from_step = []

for i in range(5):

    u = solve_with_method_1(u, x, t, v, dt, dx, nx, alpha, f, L)

    error = np.max(np.abs(u - u_exact))

    error_from_step.append(error)

    dt /= 2

    dx /= 2

    print(error, dt, dx)


#моменты времени от t = 0 до t = 5
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
for time in range(0, 6):
    plt.plot(x, exact_solve(x, time), label=f't={time}')
plt.xlabel('x')
plt.ylabel('u_exact')
plt.title('Проекция точного решения в разные моменты времени')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
times = [0, 1, 2, 3, 4, 5]
for time in (times):
    #найдем моменты времени на сетке, максимально соответствующие t = 0 до t=5
    index = np.argmin(np.abs(time - t))
    plt.plot(x, u[index, :], label=f't={t[index]}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Проекция приближенного решения в разные моменты времени')
plt.grid(True)
plt.legend()

plt.show()

