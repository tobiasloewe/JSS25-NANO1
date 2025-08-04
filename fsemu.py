import numpy as np

def apply_1q_gate_simples(state, gate):
    state = np.asarray(state)
    gate = np.asarray(gate)
    new_state = np.empty_like(state)
    
    idim, jdim, kdim, ldim = state.shape
    for i in range(idim):
        for j in range(jdim):
            for k in range(kdim):
                for l in range(ldim):
                    new_state[i, j, k, l] = state[i, j, k, :] @ gate[:, l]
    return new_state
    

def apply_1q_gate_simple(state, gate):
    state = np.asarray(state)
    gate = np.asarray(gate)
    new_state = np.empty_like(state)
    
    for ijkl in np.ndindex(*state.shape):
        *ijk, l = ijkl
        new_state[ijkl] = state[*ijk, :] @ gate[:, l]
        
    return new_state
        

def apply_gate(state, targets, gate):
    """
    Apply a gate to the specified targets in the state.
    Parameters
    ----------
    state : array_like
        The state vector to which the gate is applied.
    targets : list of int
        The indices of the qubits to which the gate is applied.
    gate : array_like
        The gate to be applied, which should have a shape compatible with the targets.
    """
    state = np.asarray(state)
    gate = np.asarray(gate)
    
    n = state.ndim
    m = len(targets)
    
    state = np.tensordot(state, gate, (targets, np.arange(m)))

    return np.moveaxis(state, np.arange(n-m,n), targets)
    