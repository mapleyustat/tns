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

"""
with open("F_T_H.dat", "r") as f:
    i = 0
    for line in f:
        fields = line.split(" ")
        if i < NT:
            T[i] = float(fields[1])
        if i % NT == 0:
            H[i/NT] = float(fields[0])
        x[i/NT,i%NT] = float(fields[2])
        i += 1

for i in range(NH):
    plt.plot(T, x[i], marker="x", label="$H = " + str(H[i]) + "$")
plt.grid(True)
plt.title("$N_v = " + str(Nv) + "$, $N_h = " + str(Nh) + "$")
plt.legend(loc=2)
plt.xlabel("$T$ $[J]$")
plt.ylabel("$F$ $[J]$")
plt.savefig("F_T_H.png")
"""

with open("m_T_H.dat", "r") as f:
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
plt.legend(loc=1)
plt.xlabel("$T$ $[J]$")
plt.ylabel("$m = \\langle \\sigma_z^{(i)} \\rangle$")
plt.ylim(0, 1)
plt.savefig("m_T_H.png")

"""
with open("chi_T_H.dat", "r") as f:
    i = 0
    for line in f:
        fields = line.split(" ")
        x[i/NT,i%NT] = float(fields[2])
        i += 1

plt.clf()
for i in range(NH):
    plt.plot(T, x[i], marker="x", label="$H = " + str(H[i]) + "$")
plt.grid(True)
plt.title("$N_v = " + str(Nv) + "$, $N_h = " + str(Nh) + "$")
plt.legend(loc=1)
plt.xlabel("$T$ $[J]$")
plt.ylabel("$\\chi = \\frac{\partial m}{\partial H} = \\frac{1}{N} \\, \\frac{\partial^2 F}{\partial H^2}$ $[1/J]$")
plt.ylim(0, 3.5)
plt.savefig("chi_T_H.png")
"""
