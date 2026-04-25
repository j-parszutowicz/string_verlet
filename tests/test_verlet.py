import numpy as np
from string_verlet import Verlet

def test_verlet_shapes_and_initial_condition():
    n = 50
    t_max = 1.0
    dt = 0.01
    
    f0 = lambda x: np.full_like(x, 0.5)
    
    X, T, Z = Verlet(n=n, dx=0.02, dt=dt, t_max=t_max, f0=f0)
    
    assert len(X) == n
    
    oczekiwana_liczba_krokow = int(t_max / dt)
    assert Z.shape == (oczekiwana_liczba_krokow, n)
    
    np.testing.assert_allclose(Z[0, 1:-1], 0.5)
    
    assert Z[0, 0] == 0.0
    assert Z[0, -1] == 0.0