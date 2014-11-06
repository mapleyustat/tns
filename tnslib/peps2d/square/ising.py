# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:54:54 2014

@author: michael
"""

import numpy as np
from tnslib import util
from tnslib.peps2d.square import BC

def _hMatrix(H, Ni, Nj):
    h = np.ndarray((2, 2))
    h[0, 0] = 2 * H * (1.0 / Ni + 1.0 / Nj) 
    h[0, 1] = 2 + 2.0 * H / Nj
    h[1, 0] = 2 + 2.0 * H / Ni
    h[1, 1] = 0
    return h

def _twoParticleGate(T, H, Ni, Nj):
    if T == 0:
        h = np.ndarray((2, 2))
        h[0, 0] = 1 if H == 0 else 0
        h[0, 1] = h[1, 0] = 0
        h[1, 1] = 1
    else:
        h = np.exp(-0.5 * _hMatrix(H, Ni, Nj) / T)
    return h

class _Derivative:
    def __init__(self, n, order, delta):
        if n < 1:
            raise ValueError("Can not determine n-th derivative with n smaller 1!")
        if n > 2:
            raise NotImplementedError("Only first and second derivative implemented!")
        self.n = n
        if order < 1:
            raise ValueError("Can not determine derivative of orders smaller than 1!")
        if order > 2:
            self.order = 2
        else:
            self.order = order
        
        if self.n == 1:
            if self.order == 1:
                self.requiredIndices = [0, 1]
            else: # self.order == 2
                self.requiredIndices = [-1, 0, +1]
        else: # self.n == 2
            if self.order == 1:
                self.requiredIndices = [0, 1, 2]
            else: # self.order == 2
                self.requiredIndices = [-1, 0, +1]
        self.data = np.ndarray(self.requiredIndices[-1] - self.requiredIndices[0] + 1)
        self.delta = delta
    def evaluate(self, n):
        if n > self.n:
            raise ValueError("Class not instantiated for " + str(n) + "-th derivative!")
        if n == 0:
            return self.data[0]
        elif n == 1:
            if self.order == 1:
                return (self.data[1] - self.data[0]) / self.delta
            else: # self.order == 2
                return 0.5 * (self.data[1] - self.data[-1]) / self.delta
        elif n == 2:
            if self.order == 1:
                return (self.data[2] - 2.0 * self.data[1] + self.data[0]) / self.delta**2
            else:# self.order == 2
                return (self.data[1] - 2.0 * self.data[0] + self.data[-1]) / self.delta**2

class State:
    def __init__(self, T, H, BCv, BCh, Nv, Nh):
        self.p = 2
        self.D = 2
        self.H = H
        self.BCv = BCv
        self.BCh = BCh
        self.T = T
        
        if BCv != BC.infinite:
            self.Nv = Nv
        if BCh != BC.infinite:
            self.Nh = Nh
            if BCv != BC.infinite:
                self.N = Nv * Nh
        
        if BCv == BC.periodicBounds and BCh == BC.openBounds:
            if Nh < 2:
                raise ValueError("Nh >= 2 is required! (Nh = " + str(Nh) + ")")
            if Nh >= 2:
                h = _twoParticleGate(self.T, self.H, 3, 3)
                vals, vecs = np.linalg.eigh(h)
                d = np.dot(vecs, np.diag(np.sqrt(vals)))
            if Nh >= 3:
                h = _twoParticleGate(self.T, self.H, 4, 3)
                # note: this h is not symmetric! the first index stands for the
                # spin of a 4-NN-spin and the second index stands for the spin
                # of a 3-NN-spin
                
                # SVD
                #u, s, v = np.linalg.svd(h)
                #s = np.sqrt(s)
                #c1 = np.dot(u, np.diag(s))
                #c2 = np.transpose(np.dot(np.diag(s), v))
                
                # QR-decomposition
                #c1, c2 = np.linalg.qr(h)
                #c2 = np.transpose(c2)
                
                # trivial decomposition
                c1 = h
                c2 = np.identity(h.shape[0])
                
                #vals, vecs = np.linalg.eigh(h)
                #vals = np.sqrt(vals)
                #c1 = c2 = np.dot(vecs, np.diag(np.sqrt(vals)))
                
                self.r = np.fromfunction(
                    lambda s,j,k,l: d[s,j]*c2[s,k]*d[s,l], [2]*4, dtype=int)
                
            if Nh >= 4:
                h = _twoParticleGate(self.T, self.H, 4, 4)
                vals, vecs = np.linalg.eigh(h)
                b = np.dot(vecs, np.diag(np.sqrt(vals)))
                self.a = np.fromfunction(
                    lambda s,j,k,l,m: b[s,j]*b[s,k]*b[s,l]*b[s,m], [2]*5, dtype=int)
                self.b = np.fromfunction(
                    lambda s,j,k,l,m: b[s,j]*b[s,k]*b[s,l]*c1[s,m], [2]*5, dtype=int)
            if Nh == 2:
                self.r = np.fromfunction(
                    lambda s,j,k,l: d[s,j]*d[s,k]*d[s,l], [2]*4, dtype=int)
            if Nh == 3:
                self.b = np.fromfunction(
                    lambda s,j,k,l,m: b[s,j]*c1[s,k]*b[s,l]*c1[s,m], [2]*5, dtype=int)
            
        else:
            raise ValueError("Ising-PEPS for these boundary conditions is not yet implemented!")
        
    def squareModulus(self):
        if self.BCv == BC.periodicBounds and self.BCh == BC.openBounds:
            if self.Nh >= 5:
                ringA = util.buildRingMatrix(util.contractPhysicalBond(self.a), self.Nv)
            if self.Nh >= 3:
                ringB = util.buildRingMatrix(util.contractPhysicalBond(self.b), self.Nv)
            ringR = util.buildRingVector(util.contractPhysicalBond(self.r), self.Nv)
            if self.Nh == 2:
                return np.dot(ringR, ringR)
            elif self.Nh == 3:
                return np.tensordot(np.tensordot(ringR, ringB, (0, 1)), ringR, (0, 0))
            elif self.Nh == 4:
                v = np.tensordot(ringR, ringB, (0, 1))
                return np.dot(v, v)
            else:
                v = np.tensordot(ringR, ringB, (0, 1))
                m = np.linalg.matrix_power(ringA, self.Nh-4)
                return np.tensordot(np.tensordot(v, m, (0, 0)), v, (0, 0))
        else:
            raise ValueError("Not yet implemented for this BC!")
        
    def freeEnergy(self):
        return np.log(self.squareModulus()) * self.T
    
    def magnetisationThermodynamic(self, deltaH = 1e-5, order=2, returns="mF"):
        d = _Derivative(1, order, deltaH)
        for i in d.requiredIndices:
            d.data[i] = State(self.T, self.H + i*deltaH, self.BCv, self.BCh, self.Nv, self.Nh).freeEnergy()
        if returns == "m":
            return d.evaluate(1) / self.N + 1.0
        else: # returns == "mF"
            return d.evaluate(1) / self.N + 1.0, d.evaluate(0)
    
    def susceptibilityThermodynamic(self, deltaH = 1e-5, order=2, returns="XmF"):
        d = _Derivative(2, order, deltaH)
        for i in d.requiredIndices:
            d.data[i] = State(self.T, self.H + i*deltaH, self.BCv, self.BCh, self.Nv, self.Nh).freeEnergy()
        if returns == "X":
            return d.evaluate(2) / self.N
        elif returns == "Xm":
            return d.evaluate(2) / self.N, d.evaluate(1) / self.N + 1.0
        elif returns == "XF":
            return d.evaluate(2) / self.N, d.evaluate(0)
        else: # returns == "XmF"
            return d.evaluate(2) / self.N, d.evaluate(1) / self.N + 1.0, d.evaluate(0)
    
    def oneBodyObservableInnermost(self, o):
        if self.BCv == BC.periodicBounds and self.BCh == BC.openBounds:
            if self.Nh >= 5:
                a2 = util.contractPhysicalBond(self.a)
                o2 = np.tensordot(np.tensordot(o, self.a, (0, 0)), np.conj(self.a), (0, 0))
                o2 = util.reindexContractedTensor(o2)
                ringA = util.buildRingMatrix(a2, self.Nv - 1)
                ringO = util.addChainLinkMatrix(ringA, o2)
                ringA = util.addChainLinkMatrix(ringA, a2)
            if self.Nh >= 3:
                b2 = util.contractPhysicalBond(self.b)
                ringB = util.buildRingMatrix(b2, self.Nv - 1)
            r2 = util.contractPhysicalBond(self.r)
            ringR = util.buildRingVector(r2, self.Nv - 1)
            if self.Nh == 2:
                o2 = np.tensordot(np.tensordot(o, self.r, (0, 0)), np.conj(self.r), (0, 0))
                o2 = util.reindexContractedTensor(o2)
                ringO = util.addChainLinkVector(ringR, o2)
                ringR = util.addChainLinkVector(ringR, r2)
                return np.dot(ringO, ringR)
            elif self.Nh == 3:
                o2 = np.tensordot(np.tensordot(o, self.b, (0, 0)), np.conj(self.b), (0, 0))
                o2 = util.reindexContractedTensor(o2)
                ringB = util.addChainLinkMatrix(ringB, o2)
                ringR = util.addChainLinkVector(ringR, r2)
                return np.tensordot(np.tensordot(ringR, ringB, (0, 1)), ringR, (0, 0))
            elif self.Nh == 4:
                o2 = np.tensordot(np.tensordot(o, self.b, (0, 0)), np.conj(self.b), (0, 0))
                o2 = util.reindexContractedTensor(o2)
                ringB = util.addChainLinkMatrix(ringB, b2)
                ringR = util.addChainLinkVector(ringR, r2)
                return np.dot(
                    np.tensordot(ringR, util.addChainLinkMatrix(ringB, o2), (0, 1)),
                    np.tensordot(util.addChainLinkMatrix(ringB, b2), ringR, (0, 0))
                )
            else:
                v = np.tensordot(util.addChainLinkVector(ringR, r2), util.addChainLinkMatrix(ringB, b2), (0, 1))
                if self.Nh == 5:
                    return np.tensordot(np.tensordot(v, ringO, (0, 1)), v, (0, 0))
                if self.Nh == 6:
                    return np.tensordot(np.tensordot(np.tensordot(v, ringO, (0, 1)), ringA, (0, 1)), v, (0, 0))
                else:
                    if self.Nh % 2 == 0:
                        ringO = np.tensordot(ringO, ringA, (0, 1))
                    ringA = np.linalg.matrix_power(ringA, (self.Nh-5)/2)
                    ringA = np.tensordot(v, ringA, (0, 1))
                    return np.tensordot(np.tensordot(ringA, ringO, (0, 1)), ringA, (0, 0))
        else:
            raise ValueError("Not yet implemented for this BC!")
    
    def magnetisationInnermost(self):
        return self.oneBodyObservableInnermost(np.array([-1.0, 0.0], [0.0, 1.0]))
    """
    def oneBodyObservable(self, o, row=0, col=0):
        if row < 0 or row >= self.Nv:
            raise ValueError("The selected row " + str(row) + " is not available (Nv = " + str(self.Nv) + ")!")
        if col < 0 or col >= self.Nv:
            raise ValueError("The selected column " + str(col) + " is not available (Nh = " + str(self.Nh) + ")!")
            
        if self.BCs == BC.hopen_vperiodic:
            if self.Nh >= 5:
                a2 = util.contractPhysicalBond(self.a)
                ringA = util.buildRingMatrix(a2, self.Nv, 0, 2)
                if col >= 2 and col <= self.Nh - 3:
                    o2 = tdot(tdot(o, self.a, (0, 0)), np.conj(self.a), (0, 0))
                    o2 = util.reindexContractedTensor(o2)
                    ringO = util.buildChainMatrix(a2, self.Nv - 1, 0, 2)
            if self.Nh >= 3:
                b2 = util.contractPhysicalBond(self.b)
                ringB = util.buildRingMatrix(b2, self.Nv, 0, 2)
                if col == 1 or col == self.Nh - 2:
                    o2 = tdot(tdot(o, self.b, (0, 0)), np.conj(self.b), (0, 0))
                    o2 = util.reindexContractedTensor(o2)
                    ringO = util.buildChainMatrix(b2, self.Nv - 1, 0, 2)
            if col == 0 or col == self.Nh - 1:
                o2 = tdot(tdot(o, self.r, (0, 0)), np.conj(self.r), (0, 0))
                o2 = util.reindexContractedTensor(o2)
                ringO = util.buildChainMatrix(r2, self.Nv - 1, 0, 2)
            r2 = util.contractPhysicalBond(self.r)
            ringR = util.buildRingMatrix(r2, self.Nv, 0, 2)
            ringO = tdot(ringO, o2, (2, 0))
            if len(ringO.shape) == 6:
                ringO = tsum(ringO, (0, 1, 3, 2, 0, 4))
            else:
                ringO = tsum(ringO, (0, 1, 2, 0))
            ringO = ringO.reshape(ringO.shape[0]*ringO.shape[1], ringO.shape[0]*ringO.shape[1])
            if self.Nh == 2:
                return np.dot(ringR, ringO)
            if self.Nh == 3:
                if col == 1:
                    return tdot(tdot(ringR, ringO, (0, 1)), ringR, (0, 0))
                else:
                    return tdot(tdot(ringO, ringB, (0, 1)), ringR, (0, 0))
            else:
                if col == 0 or col == self.Nh - 1:
                    return -1
                elif col == 1 or col == self.Nh - 2:
                    return -1
                else:
                    return -1
        else:
            raise ValueError("Not yet implemented for this BC!")
    
    def oneBodyObservableSumSites(self, o):
        return 0
    def magnetisation(self):
        return self.oneBodyObservableSumSites(np.array([-1.0, 0.0], [0.0, 1.0]))
    def magnetisationDensity(self):
        return self.oneBodyObservableSumSites(np.array([-1.0, 0.0], [0.0, 1.0])) / (self.N)
    def magnetisationSingleSite(self):
        return self.oneBodyObservable(np.array([-1.0, 0.0], [0.0, 1.0]), self.Nh / 2)
    """

def create(T, H, BCv, BCh, Nv, Nh):
    """Creates the elementary tensors for an Ising PEPS with periodic boundary 
    conditions in one direction and open boundary conditions in the orthogonal
    direction.
    
    Parameters
    ----------
    T    : double
           The temperature of the system in units of the coupling strength.
    H    : double
           The external field in units of the coupling strength.
    BCv  : int
           The boundary conditions for the vertical direction. Use the values 
           from tnslib.peps2d.square.BC.
    BCh  : int
           The boundary conditions for the horizontal direction. Use the values 
           from tnslib.peps2d.square.BC.  
    Nv   : int
           The number of lattice sites in the vertical direction.
           If an infinte lattice is chosen via the BCs, tihs parameter is 
           ignored.
    Nh   : int
           The number of lattice sites in the horizontal direction.
           If an infinte lattice is chosen via the BCs, tihs parameter is 
           ignored.
    
    Returns
    -------
    tns : tnslib.peps2d.square.ising.State
          The tensor network state.
    """
    return State(T, H, BCv, BCh, Nv, Nh)
