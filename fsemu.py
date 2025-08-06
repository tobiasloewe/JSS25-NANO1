import numpy as np
import itertools

gates = {'X': np.array([[0, 1], [1, 0]]),
         'Y': np.array([[0, -1j], [1j, 0]]),
         'Z': np.array([[1, 0], [0, -1]]),
         'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
         'CNOT': np.array([[[[1, 0], [0, 0]],[[0, 1],[0, 0]]],[[[0, 0],[0, 1]],[[0, 0],[1, 0]]]])
        }

def sample_state(state):
    P = state * np.conjugate(state)
    r = np.random.random()
    tmp = 0
    for s in np.ndindex(P.shape):
        tmp += P[s]
        if tmp > r:
            return s
    return s

def get_zexpect(state, index):
    state1 = apply_gate(state, (index,), 'Z')
    state1 = state1 * np.conjugate(state)
    return np.sum(state1)

def get_zcorrelator(state, index1, index2):
    state1 = apply_gate(state, (index1,), 'Z')
    state1 = apply_gate(state1, (index2,), 'Z')
    state1 = state1 * np.conjugate(state)
    return np.sum(state1)

def expectation_value(state, operations):
    state1 = simulate(state, operations)
    state1 = state1*np.conjugate(state)
    return np.sum(state1)
    

def get_probabilities(state):
    state = state.flatten()
    n = np.log2(len(state))
    probabilities = np.abs(state) ** 2
    basis  = [''.join(bits) for bits in itertools.product('01', repeat=int(n))]
    return dict(zip(basis, probabilities))
    
def init_state(n):
    quantum_state = np.zeros((2,) * n)
    quantum_state[(0,) * n] = 1
    return quantum_state       

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
    if type(gate) is str:
        gate = gates[gate]
    
    state = np.asarray(state)
    gate = np.asarray(gate)
    
    n = state.ndim
    m = len(targets)
    
    state = np.tensordot(state, gate, (targets, np.arange(m)))

    return np.moveaxis(state, np.arange(n-m,n), targets)

def simulate(state, gates):
    
    if type(state) is int:
        state = init_state(state)
        
    for gate, targets in gates:
        state = apply_gate(state, targets, gate)
    return state
    

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

    