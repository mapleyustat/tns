# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:35:02 2014

@author: michael
"""

import os
import sys
import tnslib.peps2d.square.ising
from tnslib.peps2d.square import BC
import numpy as np
import time

if len(sys.argv) < 3:
    print "usage: python peps2d_square_ising_m_T_H.py x numProcs"
    print "  note: number of T-points is 7 + 6*x"
    print "  note: will fork numProcs processes"
    exit(-1)
    
startTime = time.time()

Nv = 5
Nh = [5, 10, 20, 40]
NNh = len(Nh)

NT = 7 + int(sys.argv[1])*6
T = np.linspace(0, 6, NT)
H = 0.1

numProcs = int(sys.argv[2])
numDataPointsToProc = -(-(NT*NNh) / numProcs) # int-division with ceil

os.system("mkdir -p tmp")
os.system("rm tmp/F_T_N_*.dat")
os.system("rm tmp/m_T_N_*.dat")
os.system("rm tmp/chi_T_N_*.dat")

for i in range(numProcs):
    if os.fork() == 0:
        firstDataPoint = i*numDataPointsToProc
        lastDataPoint = (i+1)*numDataPointsToProc
        if lastDataPoint > NT * NNh:
            lastDataPoint = NT * NNh
        if lastDataPoint <= firstDataPoint:
            print "nothing left for child process", i
            exit()
        
        print "child process", i, "doing", (lastDataPoint-firstDataPoint), "data points"
        
        outpidsuffix = "_" + str(i).zfill(2)
        f1 = open("tmp/F_T_H" + outpidsuffix + ".dat", "w")
        f2 = open("tmp/m_T_H" + outpidsuffix + ".dat", "w")
        f3 = open("tmp/chi_T_H" + outpidsuffix + ".dat", "w")
        
        for j in range(firstDataPoint, lastDataPoint):
            tns = tnslib.peps2d.square.ising.create(T[j%NT], H, BC.periodicBounds, BC.openBounds, Nv, Nh[j/NT])
            chi, m, F = tns.susceptibilityThermodynamic()
            f1.write(str(Nh[j/NT]) + " " + str(T[j%NT]) + " " + str(F) + "\n")
            f2.write(str(Nh[j/NT]) + " " + str(T[j%NT]) + " " + str(m) + "\n")
            f3.write(str(Nh[j/NT]) + " " + str(T[j%NT]) + " " + str(chi) + "\n")
        
        f1.close()
        f2.close()
        f3.close()
        exit()

for i in range(numProcs):
    os.wait()

print "runtime:", time.time() - startTime, "seconds"

os.system("cat tmp/F_T_H_*.dat > F_T_H.dat")
os.system("cat tmp/m_T_H_*.dat > m_T_H.dat")
os.system("cat tmp/chi_T_H_*.dat > chi_T_H.dat")
os.system("python plot.py " + str(Nv) + " " + str(H) + " " + str(NT) + " " + str(NNh))
exit()
