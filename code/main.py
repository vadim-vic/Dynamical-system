import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz system of ODEs
def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial condition and time span
y0 = [1.0, 1.0, 1.0]
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Integrate the system using solve_ivp with RK45 (adaptive Runge-Kutta)
solution = solve_ivp(lorenz, t_span, y0, t_eval=t_eval, method='RK45')

# Extract results
x, y, z = solution.y

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("Lorenz Attractor (SciPy RK45)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
