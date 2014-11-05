# -*- coding: utf-8 -*-

import numpy as np
from numpy import tensordot as tdot
from numpy import einsum as tsum

def reindexContractedTensor(a):
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
    return np.reshape(a, [D**2]*n)

def contractPhysicalBond(a):
    """ Performs the contraction b = sum_i a[i,j1,k1,...] a*[i,j2,k2,...],
    rearanges the indics, such that one has b[j1,j2,k1,k2,...] and combines
    the indices, such that result is b[j,k,...]. If an initial index j1 was 
    of dimension D, the resulting index j is of dimension D**2.
    """
    return reindexContractedTensor(tdot(a, np.conj(a), (0, 0)))

#def buildChain(a, n, i, j):
    """ Build a chain, where each link is a tensor a. The indices i and j of 
    two adjacent tensors are connected. The first index of the resulting tensor
    is the initial index i of the first tensor a in the chain and the last 
    index of the resulting tensor is the index j of the last tensor a in the 
    chain. The other indices are ordered such that the first (j-i-1)*n indices
    are the indices between i and j of all the n tensors in the chain. The
    rear indices are the remaining indices.
    Example: a is a rank-4 tensor; i=0; j=2; n=3; The result is
    b[i,j1,j2,j3,k1,k2,k3,l] = 
      sum_{m1,m2} a[i,j1,m1,k1]*a[m1,j2,m2,k2]*a[m2,j3,l,k3].
    Mind the order of the indices of the result b!
    """
    """
    if i == j:
        raise ValueError("Can't contract equal indices to a chain!")
    if i > j:
        i, j = j, i
    m = len(a.shape)
    r = a
    for k in range(n-1):
        r = tdot(r, a, (j+k*(m-2), i))
        
    # roll index i of the first tensor to the front
    r = np.rollaxis(r, i, 0)
    # roll index j of last tensor to the back
    r = np.rollaxis(r, j+(n-1)*(m-2), n*(m-2)+1)
    
    # roll the indices between index i and j of each tensor to the front
    p = j-i-1
    for k in range(n):
        for l in range(p):
            r = np.rollaxis(r, k*(m-2)+(i+l+1), l+k*p+1)
    return r
    """
"""
def buildRing(a, n, i, j):
    chain = buildChain(a, n, i, j)
    m = len(chain.shape)
    return tsum(chain, map(lambda x: x % (m-1), range(m)))
"""

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
    raise NotImplementedError("Not yet implemented!")

def buildRing(a, n, i, j):
    # reshape first and then use buildRing{Matrix/Vector}
    raise NotImplementedError("Not yet implemented!")

"""
def buildChainMatrix(a, n, i, j):
    d1 = 1
    for k in range(i+1, j):
        d1 *= a.shape[k]
    d2 = 1
    for k in range(i):
        d2 *= a.shape[k]
    for k in range(j+1, len(a.shape)):
        d2 *= a.shape[k]
    
    chain = buildChain(a, n, i, j)
    
    newshape = [ chain.shape[0] ]
    if d1 > 1:
        newshape.append(d1**n)
    if d2 > 1:
        newshape.append(d2**n)
    
    newshape.append(chain.shape[-1])
    return chain.reshape(newshape)
"""

"""
def buildRingMatrix(a, n, i, j):
    if i == 0 and j == 2:
        return buildRingMatrix02(a, n)
    else:
        raise NotImplementedError("Only implemented for i=0 and j=2!")
"""
"""
    chain = buildChainMatrix(a, n, i, j)
    if (len(chain.shape) == 3):
        return tsum(chain, (0, 1, 0))
    else: # len(chain.shape) == 4
        return tsum(chain, (0, 1, 2, 0))
"""
def buildRingMatrix02(a, n):
    D = a.shape[0]
    if list(a.shape) != [D]*3 and list(a.shape) != [D]*4:
        raise NotImplementedError("Only implemented for rank-3 and rank-4 tensors!")
    
    ring = a
    if len(a.shape) == 4:
        for i in range(n-1):
            ring = tdot(ring, a, (2+2*i, 0))
        ring = ring.swapaxes(n*2, n*2+1)
        for i in range(n-1):
            ring = np.rollaxis(ring, 3+2*i, 2+i)
        ring = ring.reshape(D, D**n, D**n, D)
        ring = tsum(ring, (0, 1, 2, 0))
    elif len(a.shape) == 3:
        for i in range(n-1):
            ring = tdot(ring, a, (2+i, 0))
        ring = ring.reshape(D, D**n, D)
        ring = tsum(ring, (0, 1, 0))
    return ring