# -*- coding: utf-8 -*-

import numpy as np

def reindexContractedTensor(a, mode="reshape"):
    """ When contracting two tensors -- e.g. 
    b[k1,l1,...,k2,l2,...] = a[j,k1,l1,...] * a[j,k2,l2,...] -- one often wants
    to reorder the indices, such that one has b[k1, k2, l1, l2, ...] and 
    further combine the related indices, such that b[k, l, ...] remains.
    If the indices k1, k2 ranged from 0 to D-1, the combined index k ranges
    from 0 to D**2-1.
    """
    n = len(a.shape) / 2
    D = a.shape[0]
    
    for i in range(0, n-1):
        a = np.rollaxis(a, n+i, 2*i+1)
    
    if mode == "reshape":
        return np.reshape(a, [D**2]*n)
    elif mode == "noreshape":
        return a
    else:
        raise ValueError("Unknown mode '" + str(mode) + "'")

def contractPhysicalBond(a, mode="reshape"):
    """ Performs the contraction b = sum_i a[i,j1,k1,...] a*[i,j2,k2,...],
    rearanges the indics, such that one has b[j1,j2,k1,k2,...] and combines
    the indices, such that result is b[j,k,...]. If an initial index j1 was 
    of dimension D, the resulting index j is of dimension D**2.
    """
    return reindexContractedTensor(np.tensordot(a, np.conj(a), (0, 0)), mode)

def addChainLinkVector(chain, a):
    return np.tensordot(chain, a, (2, 0)).reshape(a.shape[0], chain.shape[1] * a.shape[1], a.shape[2])

def buildChainVector(a, n):
    if len(a.shape) != 3:
        raise ValueError("This function can only deal with rank-3 tensors!")
    D0, D1, D2 = a.shape
    F1 = D1
    b = a
    for i in range(n-1):
        F1 *= D1
        b = np.tensordot(b, a, (2, 0)).reshape(D0, F1, D2)
    return b
    
def buildRingVector(a, n):
    return np.einsum(buildChainVector(a, n), [0, 1, 0])

def addChainLinkMatrix(chain, a):
    return np.tensordot(chain, a, (2, 0)).swapaxes(2, 3).swapaxes(3, 4).reshape(a.shape[0], chain.shape[1] * a.shape[1], a.shape[2], chain.shape[3] * a.shape[3])

def buildChainMatrix(a, n):
    if len(a.shape) != 4:
        raise ValueError("This function can only deal with rank-4 tensors!")
    D0, D1, D2, D3 = a.shape
    F1, F3 = D1, D3
    b = a
    for i in range(n-1):
        F1 *= D1
        F3 *= D3
        b = np.tensordot(b, a, (2, 0)).swapaxes(2, 3).swapaxes(3, 4).reshape(D0, F1, D2, F3)
    return b

def buildRingMatrix(a, n):
    return np.einsum(buildChainMatrix(a, n), [0, 1, 0, 2])

def buildChain(a, n, i, j):
    # reshape first and then use buildChain{Matrix/Vector}
    """
    M = len(a.shape)
    if (i + 1) % M == j:
        a = np.rollaxis(a, i, 0)
        a = np.rollaxis(a, j, M-1)
        a = a.reshape(a.shape[0], np.prod(a.shape[1:M-1]), a.shape[-1])
        return buildChainVector(a, n)
    if (j + 1) % M == i:
        a = np.rollaxis(a, j, 0)
        a = np.rollaxis(a, i, M-1)
        a = a.reshape(a.shape[0], np.prod(a.shape[1:M-1]), a.shape[-1])
        return buildChainVector(a, n)
    if i == 0
    """
    raise NotImplementedError("Not yet implemented!")

def buildRing(a, n, i, j):
    # reshape first and then use buildRing{Matrix/Vector}
    raise NotImplementedError("Not yet implemented!")

