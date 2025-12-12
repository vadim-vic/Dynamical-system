import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


# Lorenz system
def lorenz(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])


# Simple Euler integration in JAX
#@jax.jit
def integrate_euler(f, y0, t):
    def step(y, t_next):
        dt = t_next - t[0]
        return y + dt * f(y, t_next), y + dt * f(y, t_next)

    _, ys = jax.lax.scan(step, y0, t[1:])
    ys = jnp.vstack([y0, ys])
    return ys


# Time range and initial conditions
t = jnp.linspace(0.0, 40.0, 10000)
y0 = jnp.array([1.0, 1.0, 1.0])

# Integrate
trajectory = integrate_euler(lorenz, y0, t)

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
