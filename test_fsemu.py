import numpy as np
import numpy.testing as npt
import fsemu

def test_GHZ():
    GHZ_state = np.zeros((2, 2, 2, 2))
    GHZ_state[0, 0, 0, 0] = 1 / np.sqrt(2)
    GHZ_state[1, 1, 1, 1] = 1 / np.sqrt(2)
    
    H  = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    CX = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]])

    # Reshape into (2,2,2,2)
    CX = CX.reshape(2,2,2,2)
    
    state = np.zeros((2, 2, 2, 2))
    state[0, 0, 0, 0] = 1
    
    state = fsemu.apply_gate(state, [0], H)
    state = fsemu.apply_gate(state, [0, 1], CX)
    state = fsemu.apply_gate(state, [0, 2], CX)
    state = fsemu.apply_gate(state, [0, 3], CX)
    
    npt.assert_allclose(state, GHZ_state)