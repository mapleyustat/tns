# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:29:42 2014

@author: michael
"""

import os
import sys
import tnslib.peps2d.square.ising
from tnslib.peps2d.square import BC
import numpy as np
import time

if len(sys.argv) < 3:
    print "usage: python peps2d_square_ising_sigmax_T_H.py x numProcs"
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
os.system("rm tmp/sigmax_T_N_*.dat")

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
        f2 = open("tmp/sigmax_T_N" + outpidsuffix + ".dat", "w")
        
        for j in range(firstDataPoint, lastDataPoint):
            tns = tnslib.peps2d.square.ising.create(T[j%NT], H, BC.periodicBounds, BC.openBounds, Nv, Nh[j/NT])
            s = tns.oneBodyObservableInnermost(np.array([[0.0, 1.0], [1.0, 0.0]]))
            f2.write(str(Nh[j/NT]) + " " + str(T[j%NT]) + " " + str(s) + "\n")
        
        f2.close()
        exit()

for i in range(numProcs):
    os.wait()

print "runtime:", time.time() - startTime, "seconds"

os.system("cat tmp/sigmax_T_N_*.dat > sigmax_T_N.dat")
os.system("python plot.py " + str(Nv) + " " + str(H) + " " + str(NT) + " " + str(NNh))
exit()
