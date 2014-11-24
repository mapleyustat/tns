# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt

Nv = int(sys.argv[1])
Nh = int(sys.argv[2])
NT = int(sys.argv[3])
NH = int(sys.argv[4])
T = np.ndarray(NT)
H = np.ndarray(NH)
x = np.ndarray((NH, NT))

with open("sigmax_T_H.dat", "r") as f:
    i = 0
    for line in f:
        fields = line.split(" ")
        if i < NT:
            T[i] = float(fields[1])
        if i % NT == 0:
            H[i/NT] = float(fields[0])
        x[i/NT,i%NT] = float(fields[2])
        i += 1
plt.clf()
for i in range(NH):
    plt.plot(T, x[i], marker="x", label="$H = " + str(H[i]) + "$")
plt.grid(True)
plt.title("$N_v = " + str(Nv) + "$, $N_h = " + str(Nh) + "$")
plt.legend(loc=2)
plt.xlabel("$T$ $[J]$")
plt.ylabel("$\\langle \\sigma_x^{(i)} \\rangle$")
plt.ylim(0, 1)
plt.savefig("sigmax_T_H.png")
