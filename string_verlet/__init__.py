import numpy as np
from numba import jit

@jit(nopython=True)
def _u_next_t(u, u_prev_t, a, dt):
    return 2 * u - u_prev_t + dt ** 2 * a

@jit(nopython=True)
def _d2udx2(u, u_prev_x, u_next_x, dx):
    return (u_next_x + u_prev_x - 2 * u) / dx ** 2

@jit(nopython=True)
def _verlet_core(dx, dt, t_max, U0_array):
    n = len(U0_array)
    T = np.arange(0, t_max, dt)
    Z = np.zeros((len(T), n))
    
    U_CURR = U0_array.copy()
    U_PREV = U_CURR.copy()
    U_NEXT = np.zeros(n)

    for i in range(len(T)):
        for j in range(1, n - 1):
            a = _d2udx2(U_CURR[j], U_CURR[j - 1], U_CURR[j + 1], dx)
            U_NEXT[j] = _u_next_t(U_CURR[j], U_PREV[j], a, dt)
        
        U_PREV = U_CURR.copy()
        U_CURR = U_NEXT.copy()
        Z[i] = U_CURR

    return T, Z

def Verlet(*, n=101, dx=0.01, dt=0.005, t_max=5.0, f0):
    """
    Solves the one-dimensional wave equation for a string using the Verlet method.

    Parameters
    ----------
    n : int, optional
        Number of spatial grid points (default is 101).

    dx : float, optional
        Spatial step size (default is 0.01).

    dt : float, optional
        Time step size (default is 0.005).

    t_max : float, optional
        Total simulation time (default is 5.0).

    f0 : callable
        Function representing the initial condition of the string.
        It must accept a single argument (a 1D NumPy array X).
        Example: `lambda x: np.sin(np.pi * x)`

    Returns
    -------
    X : ndarray
        1D NumPy array of length `n` containing the spatial coordinates.

    T : ndarray
        1D NumPy array containing the successive time steps of the simulation.
        
    Z : ndarray
        2D NumPy array of shape `(len(T), n)` containing the string displacements.
        Each row `Z[i]` represents the state of the string at time `T[i]`.
    """
    X = np.linspace(0, 1, n)
    U0_array = f0(X)
    T, Z = _verlet_core(dx, dt, t_max, U0_array)
    
    return X, T, Z