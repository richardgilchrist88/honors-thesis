#!/usr/bin/env python3
## author: Leo Liberti
## purpose: generation of random instances of quantile regression
## date: 220201

############################ imports #############################
import sys
import os
import math
import numpy as np
import random

######################### global params ##########################

myZero = 1e-8
myInf = 1e30

printConstraintMatrix = False
defaultQuantile = 0.2
density = 0.8

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 3:
    exit('syntax is [./quantreg.py records fields [quantile] ]')

m = int(sys.argv[1])
n = int(sys.argv[2])

tau = defaultQuantile
if len(sys.argv) >= 4:
    tau = float(sys.argv[3])

if m < 1:
    exit('quantreg needs at least one record')

if n < 2:
    exit('quantreg needs at least two fields')

if tau > 1 or tau < 0:
    exit('quantile must be in [0,1]')

# output basename
outn = "quantreg-" + str(m) + "_" + str(n)

# data table
D = np.random.uniform(-1,1, (m,n))   # in [-1,1]
# make this sparse
for i in range(m):
    for j in range(n):
        if random.uniform(0,1) > density:
            D[i,j] = 0.0

# dependent column index: last
bidx = n-1

if printConstraintMatrix:
    Id = np.eye(m)
    A = np.hstack((D[:,0:bidx], D[:,bidx+1:n], Id, -Id))
    b = D[:,bidx]

# output to AMPL .dat
dat = outn + ".dat"
with open(dat, 'w') as out:
    print("# AMPL .dat file encoding quantile regression instance", file=out)
    print("# m={0:d} n={1:d} tau={2:.3f} density={3:.2f}".format(m,n,tau,density), file=out)
    print("param m := {0:d};".format(m), file=out)
    print("param n := {0:d};".format(n), file=out)
    print("param bidx := {0:d};".format(bidx+1), file=out)
    print("param tau := {0:f};".format(tau), file=out)
    print("param D :=", file=out)
    for i in range(m):
        for j in range(n):
            if abs(D[i,j]) > myZero:
                print(" {0:d} {1:d} {2:f}".format(i+1,j+1, D[i,j]), file=out)
    print(";", file=out)
    if printConstraintMatrix:
        print("param A :=", file=out)
        for i in range(m):
            for j in range(n-1+2*m):
                if abs(A[i,j]) > myZero:
                    print(" {0:d} {1:d} {2:f}".format(i+1,j+1,A[i,j]),file=out)
        print(";", file=out)
        print("param b :=", file=out)
        for i in range(m):
            if abs(b[i]) > myZero:
                    print(" {0:d} {1:d} {2:f}".format(i+1,b[i]),file=out)
        print(";", file=out)
    
