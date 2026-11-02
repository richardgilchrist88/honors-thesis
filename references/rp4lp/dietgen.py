#!/usr/bin/env python3
## author: Leo Liberti
## purpose: generation of random diet problem instances
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

density = 0.5

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 3:
    exit('syntax is [./dietgen.py nutrients foods]')

m = int(sys.argv[1])
n = int(sys.argv[2])

if m < 1:
    exit('diet needs at least one nutrient')

if n < 1:
    exit('diet needs at least one food')

# output basename
outn = "diet-" + str(m) + "_" + str(n)

# nutrient-in-food matrix
D = np.random.rand(m,n)   # in [0,1]
# make this sparse
for i in range(m):
    for j in range(n):
        if random.uniform(0,1) > density:
            D[i,j] = 0.0
## scale rows differently (nutrients have different units)
## (wlog we can rescale rows, so this is useless)
#scale = np.array([random.randint(1,10) for i in range(m)])
#diagscale = np.diag(scale)
#D = np.dot(diagscale, D)

if printConstraintMatrix:
    Id = np.eye(m)
    A = np.hstack((D, -Id))

# nutrient needs
b = np.random.rand(m)

# food cost
c = np.random.rand(n)

# output to AMPL .dat
dat = outn + ".dat"
with open(dat, 'w') as out:
    print("# AMPL .dat file encoding diet instance", file=out)
    print("# m={0:d} n={1:d} density={2:.2f}".format(m,n,density), file=out)
    print("param m := {0:d};".format(m), file=out)
    print("param n := {0:d};".format(n), file=out)
    print("param b :=", file=out)
    for i in range(m):
        print(" {0:d} {1:f}".format(i+1,b[i]), file=out)
    print(";", file=out)
    print("param c :=", file=out)
    for i in range(n):
        print(" {0:d} {1:f}".format(i+1,c[i]), file=out)
    print(";", file=out)
    print("param D :=", file=out)
    for i in range(m):
        for j in range(n):
            if abs(D[i,j]) > myZero:
                print(" {0:d} {1:d} {2:f}".format(i+1,j+1, D[i,j]), file=out)
    print(";", file=out)
    if printConstraintMatrix:
        print("param A :=", file=out)
        for i in range(m):
            for j in range(n+m):
                if abs(A[i,j]) > myZero:
                    print(" {0:d} {1:d} {2:f}".format(i+1,j+1,A[i,j]),file=out)
        print(";", file=out)
    
