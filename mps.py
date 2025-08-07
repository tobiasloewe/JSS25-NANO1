import numpy as np
import itertools

gates = {'X': np.array([[0, 1], [1, 0]]),
         'Y': np.array([[0, -1j], [1j, 0]]),
         'Z': np.array([[1, 0], [0, -1]]),
         'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
         'CNOT': np.array([[[[1, 0], [0, 0]],[[0, 1],[0, 0]]],[[[0, 0],[0, 1]],[[0, 0],[1, 0]]]])
        }


def init_state_random(d, D):
    """
    Initialize a Matrix Product State (MPS) with random tensors
    Args:
        d (int): Physical dimension of the system
        D (int): Bond dimension of the MPS
    Returns:
        mps (list): List of tensors representing the MPS
    """
    mps = []
    for i in range(d):
        if i == 0:
            tensor = np.random.rand(1, d, D)
        elif i == d - 1:
            tensor = np.random.rand(D, d, D)
        else:
            tensor = np.random.rand(D, d, 1)
        mps.append(tensor)
    return mps

def init_state_zero(N):
    state = init_zeros(N,2)
    for i in range(N):
        state[i][0, 0, 0] = 1
    return state
    
def init_zeros(N, d):
    """
    Initialize a Matrix Product State (MPS) with zero tensors
    Args:
        N (int): Number of sites in the MPS
        D (int): physical dimension
    Returns:
        mps (list): List of tensors representing the MPS
    """
    mps = []
    
    for i in range(N):
        tensor = np.zeros((1, d, 1))
        mps.append(tensor)
    return mps

def flatten(mps):
    fullstate = as_array(mps)
    fullstate = fullstate.flatten()
    n = np.log2(len(fullstate))
    probabilities = np.abs(fullstate) ** 2
    basis  = [''.join(bits) for bits in itertools.product('01', repeat=int(n))]
    return dict(zip(basis, probabilities))

def as_array(mps):
    flat = mps[0]
    for tensor in mps[1:]:
        flat = np.tensordot(flat, tensor, 1)
        
    return flat.squeeze()

def apply_1q_gate(mps, target, gate):
    if isinstance(gate, str):
        gate = gates[gate]
    
    mps[target] = np.tensordot(mps[target], gate, axes=([1], [0]))
    mps[target] = np.moveaxis(mps[target], 2, 1)
    return mps
    
def apply_2q_gate_nn(mps, targets, gate):
    
    if isinstance(gate, str):
        gate = gates[gate]
    
    tA, tB = targets
    if tA > tB:
        tA, tB = tB, tA
        gate = np.transpose(gate, (1, 0, 3, 2))
    assert tB == tA + 1, "2-qubit gate must be applied to nearest neighbors"
    
    A = mps[tA]
    B = mps[tB]
    
    assert A.ndim == 3 and B.ndim == 3, "Tensors must be 3-dimensional"
    assert gate.ndim == 4, "Gate must be 4-dimensional"
    
    AB = np.tensordot(A, B, 1)
    AB = np.tensordot(gate, AB, ((2,3),(1,2)))
    
    AB = AB.transpose((2,0,1,3))
    
    # QR
    bond_A, out_A, out_B, bond_B = AB.shape

    AB = AB.reshape(bond_A*out_A, out_B*bond_B)
    Q,R = np.linalg.qr(AB)
    new_bond = Q.shape[1]
    
    A = Q.reshape(bond_A, out_A, new_bond)
    B = R.reshape(new_bond, out_B, bond_B)
    
    new_state = list(mps)
    new_state[tA] = A
    new_state[tB] = B
    return new_state

def left_canonicalize(mps, index):
    new_mps = list(mps)
    if index == 0:
        return new_mps
    
    for i in range(index):
        A = new_mps[i]
        B = new_mps[i + 1]
        AB = np.tensordot(A, B, 1)

        bond_A, out_A, out_B, bond_B = AB.shape

        AB = AB.reshape(bond_A*out_A, out_B*bond_B)
        Q,R = np.linalg.qr(AB)
        new_bond = Q.shape[1]

        new_mps[i] = Q.reshape(bond_A, out_A, new_bond)
        new_mps[i+1] = R.reshape(new_bond, out_B, bond_B)

def right_canonicalize(mps, index):
    new_mps = list(mps)
    if index == len(mps) - 1:
        return new_mps
    
    for i in range(n-1, index, -1):
        A = new_mps[i]
        B = new_mps[i - 1]
        AB = np.tensordot(A, B, 1)

        bond_A, out_A, out_B, bond_B = AB.shape

        AB = AB.reshape(bond_A*out_A, out_B*bond_B)
        Q,R = np.linalg.qr(AB)
        new_bond = Q.shape[1]

        new_mps[i] = Q.reshape(bond_A, out_A, new_bond)
        new_mps[i-1] = R.reshape(new_bond, out_B, bond_B)
    

def apply_2q_gate_nn_Tebd(mps, targets, gate):
    new_state = list(mps)
    
    if isinstance(gate, str):
        gate = gates[gate]
    
    tA, tB = targets
    if tA > tB:
        tA, tB = tB, tA
        gate = np.transpose(gate, (1, 0, 3, 2))
    assert tB == tA + 1, "2-qubit gate must be applied to nearest neighbors"
    
    
    A = mps[tA]
    B = mps[tB]
    
    assert A.ndim == 3 and B.ndim == 3, "Tensors must be 3-dimensional"
    assert gate.ndim == 4, "Gate must be 4-dimensional"
    
    AB = np.tensordot(A, B, 1)

    bond_A, out_A, out_B, bond_B = AB.shape

    AB = AB.reshape(bond_A*out_A, out_B*bond_B)
    Q,R = np.linalg.qr(AB)

    new_bond = Q.shape[1]
    
    new_state[tA] = Q.reshape(bond_A, out_A, new_bond)
    new_state[tB] = R.reshape(new_bond, out_B, bond_B)
    
    return new_state