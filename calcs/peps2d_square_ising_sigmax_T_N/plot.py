# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:41:31 2014

@author: michael
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

Nv = int(sys.argv[1])
H = float(sys.argv[2])
NT = int(sys.argv[3])
NNh = int(sys.argv[4])
T = np.ndarray(NT)
Nh = np.ndarray(NNh)
x = np.ndarray((NNh, NT))

with open("sigmax_T_N.dat", "r") as f:
    i = 0
    for line in f:
        fields = line.split(" ")
        if i < NT:
            T[i] = float(fields[1])
        if i % NT == 0:
            Nh[i/NT] = float(fields[0])
        x[i/NT,i%NT] = float(fields[2])
        i += 1

plt.clf()
for i in range(NNh):
    plt.plot(T, x[i], marker="x", label="$N_h = " + str(Nh[i]) + "$")
plt.grid(True)
plt.title("$N_v = " + str(Nv) + "$, $H = " + str(H) + "$")
plt.legend(loc=2)
plt.xlabel("$T$ $[J]$")
plt.ylabel("$\\langle \\sigma_x^{(i)} \\rangle$")
plt.ylim(0, 1)
plt.savefig("sigmax_T_N.png")
