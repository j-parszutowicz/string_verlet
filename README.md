# String Verlet Solver

[![Pytest CI](https://github.com/j-parszutowicz/string_verlet/actions/workflows/python-app.yml/badge.svg)](https://github.com/j-parszutowicz/string_verlet/actions/workflows/python-app.yml)

A highly optimized Python solver for the 1D wave equation using the Verlet method. Powered by **NumPy** and **Numba** for fast computations.

## Documentation
Full API documentation is available here: **[String Verlet Documentation](https://j-parszutowicz.github.io/string_verlet/)**

## Quick Start
Solving the wave equation and visualizing the results is simple. Here is an example using a Gaussian pulse as the initial condition, visualized in 3D and as a heatmap:

```python
import numpy as np
from matplotlib import pyplot as plt
from string_verlet import Verlet

# 1. Define initial string condition (Gaussian pulse)
initial_condition = lambda x: np.exp(-100 * (x - 0.5) ** 2)

# 2. Run the simulation
X, T, Z = Verlet(n=101, dx=0.01, dt=0.005, t_max=5.0, f0=initial_condition)

# 3. Prepare grids for 2D/3D plotting
X_grid, T_grid = np.meshgrid(X, T)

# --- Plot 1: 3D Surface Plot ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, T_grid, Z, cmap='viridis')
ax.set_title('3D Surface: String Evolution Over Time')
ax.set_xlabel('Position (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Displacement (u)')
plt.show()

# --- Plot 2: 2D Heatmap (pcolormesh) ---
plt.figure(figsize=(8, 5))
plt.pcolormesh(X_grid, T_grid, Z, cmap='viridis', shading='auto')
plt.colorbar(label='Displacement ($u$)')
plt.title('Heatmap: String Vibrations')
plt.xlabel('Position ($x$)')
plt.ylabel('Time ($t$)')
plt.show()
```