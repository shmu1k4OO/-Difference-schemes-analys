import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def f(x, t):
    return 2


def phi(x):
    return 2*x**2 + 1


def dphi(x):
    return 0 * np.ones_like(x)


def g_0(t):
    return 3*t**2 + 1


def g_L(t, L):
    return 2*L**2 + 3*t**2 + 1


def exact_solve(x, t):
    return 2*x**2 + 3*t**2 + 1


def solve_with_method_1(u, x, t, v, dt, dx, alpha, f, L):

    u[0, :] = phi(x)
    u[:, 0] = g_0(t)
    u[:, -1] = g_L(t, L)

    u_xx = np.zeros(len(x))
    for i in range(1, len(x)-1):
        u_xx[i] = (u[0, i-1] - 2*u[0, i] + u[0, i+1]) / dx**2

    u[1, :] = u[0, :] + dt * v + (dt**2/2) * (alpha * u_xx + f(x, 0))

    u[1, 0] = g_0(t[1])
    u[1, -1] = g_L(t[1], L)

    for j in range(1, len(t)-1):
        for i in range(1, len(x)-1):
            u[j+1, i] = (2 * u[j, i] - u[j-1, i] + 
                         dt**2 / dx**2 * alpha * (u[j, i+1] - 2*u[j, i] + u[j, i-1]) + 
                         dt**2 * f(x[i], t[j]))
    
    return u


def thomas_algorithm(A, B, C, D):
    
    n = len(D)
    
    P = np.zeros(n)
    Q = np.zeros(n)
    
    P[0] = -C[0] / B[0]
    Q[0] = D[0] / B[0]
    
    for k in range(1, n):
        denom = B[k] + A[k] * P[k-1]
        if k < n-1:
            P[k] = -C[k] / denom
        Q[k] = (D[k] - A[k] * Q[k-1]) / denom
    
    u = np.zeros(n)
    u[-1] = Q[-1]
    for k in range(n-2, -1, -1):
        u[k] = P[k] * u[k+1] + Q[k]
    
    return u


def solve_with_method_2(u, x, t, dt, dx, alpha, f, L):
    nx = len(x)
    nt = len(t)
    
    u[0, :] = phi(x)
    u[:, 0] = g_0(t)
    u[:, -1] = g_L(t, L)
    
    coefficient = alpha * dt**2 / (2 * dx**2)
    
    u_xx = np.zeros(nx)
    for i in range(1, nx-1):
        u_xx[i] = (u[0, i-1] - 2*u[0, i] + u[0, i+1]) / dx**2
    
    u[1, :] = u[0, :] + dt * dphi(x) + (dt**2/2) * (alpha * u_xx + f(x, t[0]))
    u[1, 0] = g_0(t[1])
    u[1, -1] = g_L(t[1], L)
    
    
    for j in range(1, nt-1):
        
        A = -coefficient * np.ones(nx-2)  
        B = (1 + 2*coefficient) * np.ones(nx-2)  
        C = -coefficient * np.ones(nx-2)  
        
        
        D = np.zeros(nx-2)
        for i in range(1, nx-1):
            D[i-1] = (2 * u[j, i] - u[j-1, i] + 
                     coefficient * (u[j, i+1] - 2*u[j, i] + u[j, i-1]) + 
                     dt**2 * f(x[i], t[j] + dt/2))
        
        
        D[0] -= C[0] * u[j+1, 0]  
        D[-1] -= A[-1] * u[j+1, -1]  
               
        B[0] -= C[0] * 0  
        B[-1] -= A[-1] * 0
        
        u[j+1, 1:nx-1] = thomas_algorithm(A, B, C, D)
    
    return u
    

# =====================================================
# ПАРАМЕТРЫ ЗАДАЧИ
# =====================================================


L = 1.0
T = 10 *L
alpha = 1.0

# =====================================================
# РЕШЕНИЕ НА БАЗОВОЙ СЕТКЕ
# =====================================================


nx = 191
nt = 2001

dx = L / (nx - 1)
dt = T / (nt - 1)

x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

u = np.zeros((nt, nx))
v = dphi(x)
p = phi(x)

X, T_grid = np.meshgrid(x, t)
u_exact = exact_solve(X, T_grid)
u = solve_with_method_2(u, x, t, dt, dx, alpha, f, L)
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
for time in times:
    #найдем моменты времени на сетке, максимально соответствующие t = 0 до t=5
    index = np.argmin(np.abs(time - t))
    plt.plot(x, u[index, :], label=f't={t[index]}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Проекция приближенного решения в разные моменты времени')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'figures_second/Paraboloid/stable.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()


# =====================================================
# ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТА
# =====================================================


os.makedirs('figures', exist_ok=True)

n_levels = 4

errors = np.zeros((n_levels, n_levels))

row_names = []
col_names = []

for step_tau in range(n_levels):
    if step_tau == 0:
        row_names.append('τ₀')
        col_names.append('h₀')
    else:
        row_names.append(f'τ₀/{2**step_tau}')
        col_names.append(f'h₀/{2**step_tau}')

    for step_h in range(n_levels):

        current_dt = dt / (2**step_tau)
        current_dx = dx / (2**step_h)

        nt = int(T / current_dt) + 1
        nx = int(L / current_dx) + 1

        t = np.linspace(0, T, nt)
        x = np.linspace(0, L, nx)

        v = dphi(x)

        u = np.zeros((nt, nx))
        u = solve_with_method_2(u, x, t, current_dt, current_dx, alpha, f, L)

        X, T_grid = np.meshgrid(x, t)
        u_exact = exact_solve(X, T_grid)
        error = np.max(np.abs(u - u_exact))
        errors[step_tau, step_h] = error
        flag = 2
        
        if flag == 1:
            courant = alpha * current_dt / current_dx
            if courant >= 1:

                print(f"dt^{step_tau}, dx^{step_h}: courant={courant:.3f}")
                errors[step_tau, step_h] = np.nan
                
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
                for time in times:
                    #найдем моменты времени на сетке, максимально соответствующие t = 0 до t=5
                    index = np.argmin(np.abs(time - t))
                    plt.plot(x, u[index, :], label=f't={t[index]}')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.title(f'Проекция приближенного решения в разные моменты времени при dt^2/dx^2={courant:.3f}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'figures_second/My example/unstable_dt{step_tau}_dx{step_h}.png', dpi=150, bbox_inches='tight')
                plt.show()
                plt.close()        

                print(f"dt/2^{step_tau}, dx/2^{step_h}: courant={courant:.3f}, error={error:.3f}")

        elif flag == 2:
            print(f"dt/2^{step_tau}, dx/2^{step_h}: error={error:.3f}")

    
errors_from_steps = pd.DataFrame(errors, index=row_names, columns=col_names)

print(errors_from_steps)

print("Погрешность аппроксимации:")
for i in range(1, errors.shape[0]):
    print(f"при t={i}: {errors[i-1, i-1] / errors[i, i]}")

