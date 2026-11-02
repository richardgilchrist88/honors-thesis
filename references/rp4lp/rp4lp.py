#!/usr/bin/env python3

## author: Leo Liberti
## purpose: solve org and proj instances, compare
## date: 220201

############################ imports #############################
import sys
import gzip
import io
import os
import glob
import math
import time
import random
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import itertools
from amplpy import AMPL

######################### global params ##########################

# print out more run info
debug = True

# don't solve original problem
solveOriginal = True
#solveOriginal = False ## doesn't work here

# used in writing out large .dat files
outLineBuffer = 500

myZero = 1e-9
mySmall = 1e-3
myInf = 1e30

## perform RP experiments
#jllEPS = sorted([0.15, 0.2, 0.25, 0.3, 0.35, 0.4]) #SEA22 subm
#jllEPS = sorted([0.1, 0.125, 0.15, 0.175]) #meaningful
jllEPS = sorted([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) #JEA
#jllEPS = sorted([0.05, 0.1, 0.15, 0.2]) # JEA test2 for quantreg2
#jllEPS = sorted([0.1, 0.15]) # JEA test2 for basispursuit2
runsPerEps = 5   # how many times we solve instance for each epsilon
## or just run once
jllEPS = [0.5]
runsPerEps = 1

# some other values used in sampling RP
RPDensFactor = 0.5 # set RP density at RPDensFactor * [Ax=b density]
universalConstant = 1.0

# LP structures 
instanceTypes = ["basis pursuit", "diet", "max flow", "quantile regression", "uniform"]

# python starts from zero, AMPL starts from 1
offset = 1

## solver and options
LPSolverOrg = "cplex"
cplexoptions = "autopt display=1"
orgoptions = cplexoptions
LPSolverPrj = "highs"
#LPSolverPrj = "cplex"
## barrier solver for projected problem if using the AMPL interface
prjcplexoptions = "baropt crossover=0 display=1 bardisplay=1"
highsoptions = "alg:method=ipm lim:ipmiterationlimit=10 tech:outlev=1"
prjoptions = highsoptions
#prjoptions = prjcplexoptions

# sparse or dense algebra (option only applicable to maxflow)
sparseFlag = False
sparseFlag = True

# retrieval method (choice only applicable to diet problem)
retrJLLMOR = True
retrJLLMOR = False

# coefficient of slack vars in objective (relevant for diet problem)
slackCoeff = 10.0

## alternating projection method retrieval to improve feasibility:
#   alt proj between Ax=b and l<=x<=u without solving original problem:
#   1. min ||Ax-b||_2 (pseudoinverse) and 2. cap components outside of range
#AltProjRetr = False  # don't use alternating projection post-retrieval
AltProjRetr = True   # use alternating projection post-retrieval
#AltProjOpt = True    # also try and improve optimality
AltProjOpt = False   # only limit to feasibility 
AltProjMaxItn = 30   # number of iterations for feasibility alt proj
AltProjTol = 0.01    # error tolerance for alt proj
#AltProjMethod = "AARM"       # Artacho et al's Averaged APM (AARM)
#AltProjMethod = "vonNeumann" # von Neumann's original APM
AltProjMethod = "Dykstra"    # Dykstra's APM
#AltProjMethod = "locslv"    # a few iterations with a local solver

## file names
csvName = "rp4lp.csv"

maxflowMod = "maxflow.mod"
dietMod = "diet.mod"
quantregMod = "quantreg.mod"
basispursuitMod = "basispursuit.mod"
uniformMod = "uniform.mod"

maxflowProjMod = "maxflowprj.mod"
#dietProjMod = "dietprj.mod"  # in SEA22 paper
dietProjMod = "dietprj1.mod"  # n.1 in JEA paper
#dietProjMod = "dietprj2.mod"  # n.2 in JEA paper
quantregProjMod = "quantregprj.mod"
basispursuitProjMod = "basispursuit.mod"
uniformProjMod = "uniform.mod"

basispursuitDat = "basispursuit.dat"
maxflowDat = "maxflow.dat"
dietDat = "diet.dat"
quantregDat = "quantreg.dat"
uniformDat = "uniform.dat"

basispursuitProjDat = "basispursuitprj.dat"
maxflowProjDat = "maxflowprj.dat"
dietProjDat = "dietprj.dat"
quantregProjDat = "quantregprj.dat"
uniformProjDat = "uniformprj.dat"

########################### functions ############################

# print to string buffer, to increase dat printing speed
#  from https://stackoverflow.com/questions/39823303/python3-print-to-string
def printString(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

# read AMPL data file into a list of triplets (pnames,sets,vals)
#   for syntax "param : sets : pnames := vals"
# param names are in ['name'] or [['name']] (if set being def'd) in triple[0]
def readDat(fd, types):
    p = []
    pset = None
    pname = None
    pval = None
    for line in fd:
        try:
            line = line.decode('UTF-8').strip()
        except:
            pass
        # ignore empty lines or comments
        if len(line) < 1:
            continue
        if line[0] == "#":
            for n in types:
                if n in line:
                    p.append((["type"], [], [n]))
            continue
        # line elements
        l = [le.strip() for le in line.split()]
        if l[0] == "param":
            # AMPL parameter declaration
            lparm = " ".join(l[1:])
            ll = [lp.strip() for lp in lparm.split(':')]
            pset = []
            pname = []
            pval = []
            if len(ll) == 2:
                # syntax [param pname :=]
                pname.append(ll[0])
            elif len(ll) == 3:
                # syntax [param : pname1 ... pnameN := ]
                pname.append([lp.strip() for lp in ll[1].split()])
            elif len(ll) == 4:
                # syntax [param : set1 ... setM : pname1 ... pnameN :=]
                pset.append([sp.strip() for sp in ll[1].split()])
                pname.append([lp.strip() for lp in ll[2].split()])
            if ll[-1][-1] == ';':
                # end of param declaration on the line
                pval.append([float(lp) for lp in ll[-1][0:-1].split() if lp != "="])
                p.append((pname,pset,pval))
        elif l[0] == ";":
            # end of current param declaration on a line by itself
            p.append((pname,pset,pval))
        else:
            # numerical data
            if line[-1] == ';':
                # end of current param declaration
                pval.append([float(lp.strip()) for lp in line[0:-1].split()])
            else:
                pval.append([float(lp) for lp in l])
    return p

# parse .dat file triplets for max flow
def parseMaxFlow(data):
    n = 0
    src = 0
    tgt = 0
    A = []
    u = dict()
    for (p,s,v) in data:
        assert(len(p) > 0)
        if p[0] == "n":
            n = int(round(v[0][0]))
        elif p[0] == "s":
            src = int(round(v[0][0]))
        elif p[0] == "t":
            tgt = int(round(v[0][0]))
        elif p[0][0] == "u":
            for t in v:
                i = int(round(t[0]))
                j = int(round(t[1]))
                uij = float(t[2])
                A.append((i,j))
                u[(i,j)] = uij
    return (n,src,tgt,A,u)

# parse .dat file triplets for diet
def parseDiet(data):
    m = 0
    n = 0
    b = dict()
    c = dict()
    D = dict()
    for (p,s,v) in data:
        assert(len(p) > 0)
        if p[0] == "m":
            m = int(round(v[0][0]))
        elif p[0] == "n":
            n = int(round(v[0][0]))
        elif p[0] == "D":
            for t in v:
                i = int(round(t[0]))
                j = int(round(t[1]))
                Dij = float(t[2])
                D[(i,j)] = Dij
        elif p[0] == "b":
            for t in v:
                i = int(round(t[0]))
                bi = float(t[1])
                b[i] = bi
        elif p[0] == "c":
            for t in v:
                j = int(round(t[0]))
                cj = float(t[1])
                c[j] = cj
    return (m,n,b,c,D)

# parse .dat file triplets for diet
def parseQuantReg(data):
    m = 0
    n = 0
    bidx = 0
    tau = 0.0
    D = dict()
    for (p,s,v) in data:
        assert(len(p) > 0)
        if p[0] == "m":
            m = int(round(v[0][0]))
        elif p[0] == "n":
            n = int(round(v[0][0]))
        elif p[0] == "bidx":
            bidx = int(round(v[0][0]))
        elif p[0] == "tau":
            tau = float(v[0][0])
        elif p[0] == "D":
            for t in v:
                i = int(round(t[0]))
                j = int(round(t[1]))
                Dij = float(t[2])
                D[(i,j)] = Dij
    return (m,n,bidx,tau,D)

# parse .dat file triplets for diet
def parseBasisPursuit(data):
    m = 0
    n = 0
    spars = 0.0
    org = dict()
    enc = dict()
    A = dict()
    for (p,s,v) in data:
        assert(len(p) > 0)
        if p[0] == "m":
            m = int(round(v[0][0]))
        elif p[0] == "n":
            n = int(round(v[0][0]))
        elif p[0] == "s":
            spars = float(v[0][0])
        elif p[0] == "org":
            for t in v:
                i = int(round(t[0]))
                bi = float(t[1])
                org[i] = bi
        elif p[0] == "enc":
            for t in v:
                i = int(round(t[0]))
                bi = float(t[1])
                enc[i] = bi            
        elif p[0] == "A":
            for t in v:
                i = int(round(t[0]))
                j = int(round(t[1]))
                Aij = float(t[2])
                A[(i,j)] = Aij
    return (m,n,spars,org,enc,A)

# parse .dat file triplets for diet
def parseUniform(data):
    m = 0
    n = 0
    b = dict()
    c = dict()
    A = dict()
    for (p,s,v) in data:
        assert(len(p) > 0)
        if p[0] == "m":
            m = int(round(v[0][0]))
        elif p[0] == "n":
            n = int(round(v[0][0]))
        elif p[0] == "A":
            for t in v:
                i = int(round(t[0]))
                j = int(round(t[1]))
                Aij = float(t[2])
                A[(i,j)] = Aij
        elif p[0] == "b":
            for t in v:
                i = int(round(t[0]))
                bi = float(t[1])
                b[i] = bi
        elif p[0] == "c":
            for t in v:
                j = int(round(t[0]))
                cj = float(t[1])
                c[j] = cj
    return (m,n,b,c,A)

def outDatMaxFlow(n,s,t,A,u):
    with open(maxflowDat, "w") as out:
        print("# rp4lp: original max flow formulation instance", file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param s := {0:d};".format(s), file=out)
        print("param t := {0:d};".format(t), file=out)
        print("param : A : u :=", file=out)
        for (i,j) in A:
            print(" {0:d} {1:d} {2:f}".format(i,j,u[(i,j)]), file=out)
        print(";", file=out)
    return

def outDatDiet(m,n,b,c,D):
    with open(dietDat, "w") as out:
        print("# rp4lp: original diet formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param b :=", file=out)
        for i in b:
            print(" {0:d} {1:f}".format(i,b[i]), file=out)
        print(";", file=out)
        print("param c :=", file=out)
        for i in c:
            print(" {0:d} {1:f}".format(i,c[i]), file=out)
        print(";", file=out)
        print("param D :=", file=out)
        for (i,j) in D:
            print(" {0:d} {1:d} {2:f}".format(i,j, D[(i,j)]), file=out)
        print(";", file=out)
    return

def outDatQuantReg(m,p,bidx,tau,D):
    with open(quantregDat, "w") as out:
        print("# rp4lp: original quantile regression formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param p := {0:d};".format(p), file=out)
        print("param bidx := {0:d};".format(bidx), file=out)
        print("param tau := {0:f};".format(tau), file=out)
        print("param D :=", file=out)
        for (i,j) in D:
            print(" {0:d} {1:d} {2:f}".format(i,j, D[(i,j)]), file=out)
        print(";", file=out)
    return

def outDatBasisPursuit(m,n,s,org,enc,A):
    with open(basispursuitDat, 'w') as out:
        print("# rp4lp: original basis pursuit formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param s := {0:f};".format(s), file=out)
        print("param org :=", file=out)
        for i in org:
            print(" {0:d} {1:.1f}".format(i, org[i]), file=out)
        print(";", file=out)
        print("param enc :=", file=out)
        for i in enc:
            print(" {0:d} {1:f}".format(i, enc[i]), file=out)
        print(";", file=out)
        print("param A :=", file=out)
        for (i,j) in A:
            print(" {0:d} {1:d} {2:f}".format(i,j, A[(i,j)]), file=out)
        print(";", file=out)
    return

def outDatUniform(m,n,b,c,A):
    with open(uniformDat, "w") as out:
        print("# rp4lp: original uniform formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param b :=", file=out)
        for i in b:
            print(" {0:d} {1:f}".format(i,b[i]), file=out)
        print(";", file=out)
        print("param c :=", file=out)
        for i in c:
            print(" {0:d} {1:f}".format(i,c[i]), file=out)
        print(";", file=out)
        print("param A :=", file=out)
        for (i,j) in A:
            print(" {0:d} {1:d} {2:f}".format(i,j, A[(i,j)]), file=out)
        print(";", file=out)
    return

# generate a Gaussian rnd k x m (unscaled) matrix
def gaussian(k, m):
    return np.random.normal([0],[[1]],(k,m))

# generate a sparse Gaussian rnd k x m (unscaled) matrix with density s
def sparse_gaussian(k, m, s, fmt='csr'):
    T = scipy.sparse.random(k,m,density=s,format=fmt,data_rvs=np.random.randn)
    return T

# generate a Gaussian rnd k x m (unscaled) matrix with density s
def sparse_gaussian_full(k,m,s):
    T = gaussian(k,m)
    for i in range(k):
        for j in range(m):
            if np.random.rand() > s:
                T[i,j] = 0.0
    return T

def outProjDatMaxFlow(n,s,t,u,c,TA, AA,arc2col):
    with open(maxflowProjDat, "w") as out:
        print("# rp4lp: original max flow formulation instance", file=out)
        print("param rows := {0:d};".format(n), file=out)
        print("param cols := {0:d};".format(len(AA)), file=out)
        print("param s := {0:d};".format(s), file=out)
        print("param t := {0:d};".format(t), file=out)
        k = TA.shape[0]
        print("param kmax := {0:d};".format(k), file=out)
        tout0 = time.time()

        # out TA
        if debug:
            print("rp4lp:outProjDatMaxFlow: out TA... ", end='')
        print("param TA :=", file=out)
        # sparseFlag specified in Globals section
        if sparseFlag:
            TAcoo = TA.tocoo()
            for i,j,v in itertools.zip_longest(TAcoo.row, TAcoo.col, TAcoo.data):
                print(" {0:d} {1:d} {2:f}".format(i+offset, j+offset, v), file=out)
        else:
            # ## output 2D array as dict
            # for i in range(k):
            #     for j in range(len(AA)):
            #         if abs(TA[i,j]) > myZero:
            #             print(" {0:d} {1:d} {2:f}".format(i+offset, j+offset, TA[i,j]), file=out)
            # ## output 2D array in tabular form
            # for i in range(k):
            #     print(" [{0:d},*]".format(i+offset), end='', file=out)
            #     for j in range(len(AA)):
            #         if abs(TA[i,j]) > myZero:
            #             print(" {0:d} {1:f}".format(j+offset, TA[i,j]), end='', file=out)
            #     print('', file=out)
            ## output 2D array in tabular form to string buffer
            for i in range(k):
                outstr = io.StringIO()
                for j in range(len(AA)):
                    if abs(TA[i,j]) > myZero:
                        print(" {0:d} {1:f}".format(j+offset, TA[i,j]), end='', file=outstr)
                line = outstr.getvalue().strip()
                outstr.close()
                if len(line) > 1:
                    print(" [{0:d},*] {1:s}".format(i+offset, line), file=out)
        # endif sparseFlag
        print(";", file=out)
        toutTA = time.time() - tout0
        # end out TA
        # out u
        if debug:
            print("{0:f}s".format(toutTA))
        tout1 = time.time()
        if debug:
            print("rp4lp:outProjDatMaxFlow: out u... ", end='')
        ulst = [u[(i,j)] for (i,j) in AA]
        print("param u :=", file=out)
        for h,uelt in enumerate(ulst):
            print(" {0:d} {1:f}".format(h+1,uelt), file=out, end='')
            if h % outLineBuffer == 0:
                print('', file=out)
        print(";", file=out)
        toutu = time.time() - tout1
        # end out u
        # out c
        if debug:
            print("{0:f}s".format(toutu))
        if debug:
            print("rp4lp:outProjDatMaxFlow: out c")
        print("param c :=", file=out)
        for j in range(len(AA)):
            if abs(c[j]) > myZero:
                print(" {0:d} {1:.1f}".format(j+offset, c[j]), file=out)
        print(";", file=out)            
    return

def outProjDatDiet(m,n,c,k,TA,Tb):
    with open(dietProjDat, "w") as out:
        print("# rp4lp: projected diet formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param k := {0:d};".format(k), file=out)
        print("param slackCoeff := {0:f};".format(slackCoeff), file=out)
        print("param Tb :=", file=out)
        for i in range(k):
            if abs(Tb[i]) > myZero:
                print(" {0:d} {1:f}".format(i+offset,Tb[i]), file=out)
        print(";", file=out)
        print("param c :=", file=out)
        for i in c:
            print(" {0:d} {1:f}".format(i,c[i]), file=out)
        print(";", file=out)
        print("param TA :=", file=out)
        TAcoo = TA.tocoo()
        for i,j,v in itertools.zip_longest(TAcoo.row, TAcoo.col, TAcoo.data):
            print(" {0:d} {1:d} {2:f}".format(i+offset, j+offset, v), file=out)
        print(";", file=out)
    return

def outProjDatQuantReg(m,p,tau,k,TA,Tb):
    with open(quantregProjDat, "w") as out:
        print("# rp4lp: projected quantile regression formulation instance", file=out)
        print("param m := {0:d};".format(m), file=out)
        print("param k := {0:d};".format(k), file=out)
        print("param p := {0:d};".format(p), file=out)
        print("param tau := {0:f};".format(tau), file=out)
        print("param Tb :=", file=out)
        for i in range(k):
            if abs(Tb[i]) > myZero:
                print(" {0:d} {1:f}".format(i+offset,Tb[i]), file=out)
        print(";", file=out)
        print("param TA :=", file=out)
        TAcoo = TA.tocoo()
        for i,j,v in itertools.zip_longest(TAcoo.row, TAcoo.col, TAcoo.data):
            print(" {0:d} {1:d} {2:f}".format(i+offset, j+offset, v), file=out)
        print(";", file=out)

        print(";", file=out)
    return

def outProjDatBasisPursuit(k,n,s,org,TA,Tb):
    with open(basispursuitProjDat, 'w') as out:
        print("# rp4lp: original basis pursuit formulation instance", file=out)
        print("param m := {0:d};".format(k), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param s := {0:f};".format(s), file=out)
        print("param org :=", file=out)
        for i in org:
            print(" {0:d} {1:.1f}".format(i, org[i]), file=out)
        print(";", file=out)
        print("param enc :=", file=out)
        for i in range(k):
            print(" {0:d} {1:f}".format(i+offset, Tb[i]), file=out)
        print(";", file=out)
        print("param A :=", file=out)
        for i in range(k):
            for j in range(n):
                print(" {0:d} {1:d} {2:f}".format(i+offset,j+offset, TA[i,j]), file=out)
        print(";", file=out)
    return

def outProjDatUniform(k,n,c,TA,Tb):
    with open(uniformProjDat, "w") as out:
        print("# rp4lp: projected uniform formulation instance", file=out)
        print("param m := {0:d};".format(k), file=out)
        print("param n := {0:d};".format(n), file=out)
        print("param b :=", file=out)
        for i in range(k):
            if abs(Tb[i]) > myZero:
                print(" {0:d} {1:f}".format(i+offset,Tb[i]), file=out)
        print(";", file=out)
        print("param c :=", file=out)
        for i in c:
            print(" {0:d} {1:f}".format(i,c[i]), file=out)
        print(";", file=out)
        print("param A :=", file=out)
        TAcoo = TA.tocoo()
        for i,j,v in itertools.zip_longest(TAcoo.row, TAcoo.col, TAcoo.data):
            print(" {0:d} {1:d} {2:f}".format(i+offset, j+offset, v), file=out)
        print(";", file=out)
    return

def maxFlow(data, t0, jlleps, runOrg, fstar=None, xstar=None):

    if not solveOriginal:
        runOrg = False
    
    objfundir = 1.0 # maximization
    
    if not runOrg:
        assert(fstar is not None)
        assert(xstar is not None)
    
    (n,s,t,AA,u) = parseMaxFlow(data)
    xlen = len(AA) # number of arcs
    
    # density of the flow matrix (each arc yields 2 nonzeros)
    # => density = 2*|AA| / (n*|AA|) = 2/n
    # removing source+target rows shouldn't change it too much in large graphs
    constrdens = 2/n

    # some graph data structures
    arc2col = dict()
    col2arc = dict()
    for h,(i,j) in enumerate(AA):
        arc2col[(i,j)] = h
        col2arc[h] = (i,j)

    # create matrix A for Ax=0 flow constrs
    if debug:
        print("rp4lp:maxflow: creating flow matrix")
    #A = np.zeros((n-2,xlen)) #dense
    rowA = [] #sparse
    colA = [] #sparse
    valA = [] #sparse
    node2row = dict()
    row2node = dict()
    h = 0
    for i in range(offset,n+offset):
        if i not in [s,t]:
            node2row[i] = h
            row2node[h] = i
            h += 1
    for (i,j) in AA:
        if i in node2row:
            #A[node2row[i],arc2col[(i,j)]] = 1.0  #dense
            rowA.append(node2row[i])              #sparse
            colA.append(arc2col[(i,j)])           #sparse
            valA.append(1.0)                      #sparse
        if j in node2row:
            #A[node2row[j],arc2col[(i,j)]] = -1.0 #dense
            rowA.append(node2row[j])              #sparse
            colA.append(arc2col[(i,j)])           #sparse
            valA.append(-1.0)                     #sparse
    A = scipy.sparse.csr_matrix((valA, (rowA,colA)), shape=(n-2,xlen))
            
    # create objfun vector
    if debug:
        print("rp4lp:maxflow: creating objfun vector")
    c = np.zeros(xlen)
    # only objfun coeffs to be 1.0 are those indexed by (s,j)
    for (i,j) in AA:
        if i == s:
            c[arc2col[(i,j)]] = 1.0
        
    if runOrg:
        # out original instance (we could use the original .dat, but this
        #   would be unfair comparison since we're using AMPLpy in org and prj,
        #   and AMPL requires a .dat file even for prj: so we're computing the
        #   projected data in Python and outputting the projected data as .dat;
        #   this would not be the case if we used a different solver interface)
        # (not unfair if we use scipy.linprog though)
        outDatMaxFlow(n,s,t,AA,u)
        # solve original formulation with AMPLpy and CPLEX
        maxflow = AMPL()
        maxflow.setOption("solver", LPSolverOrg)
        # for default setting, comment out the two following lines
        solver_options = LPSolverOrg + "_options"
        maxflow.setOption(solver_options, orgoptions)
        maxflow.read(maxflowMod)
        maxflow.readData(maxflowDat)
        maxflow.solve()
        # get optimal objective value
        objfun = maxflow.getObjective("sourceflow")
        fstar = objfun.value()
        # get optimal solution
        xvar = maxflow.getVariable("x")
        xstar = np.zeros(xlen)
        for h,(i,j) in enumerate(AA):
            xstar[h] = xvar[i,j].value()
        # dual solution norm and theta estimate
        cons = maxflow.getConstraint("flowcons")
        dual = np.zeros(n-2)
        h = 0
        for i in range(n):
            h1 = i + offset
            if h1 not in [s,t]:
                dual[h] = cons[h1].dual()
                h += 1
        normdual = np.linalg.norm(dual)

        # # solve original formulation with scipy.linprog
        # b = np.zeros(n-2)
        # l1 = np.zeros(xlen)
        # u1 = np.array([u[col2arc[i]] for i in range(xlen)]) # UB on capacities
        # LUbnd = list(zip(l1,u1))
        # res = scipy.optimize.linprog(-c, A_eq=A, b_eq=b, bounds=LUbnd, options={"disp":True})
        # xstar = res.get('x')
        # fstar = np.dot(c,xstar)
        
    else:
        normdual = 0.0        
    thetaest = xlen
    theta = sum(xstar)        
    torg = time.time() - t0
    
            
    # sample random projector
    jlldens = RPDensFactor * constrdens
    if debug:
        print("rp4lp:maxflow: sampling RP with density={0:f}, |X|={1:d}".format(jlldens, xlen))
    #k = 3 # for testing
    k = int(round((universalConstant/jlleps)**2) * math.log(float(xlen)))
    if k > n-2:
        print("rp4lp:maxflow: {0:d} = k > n-2 = {1:d}".format(k,n-2))
    T = sparse_gaussian(k, n-2, jlldens) / math.sqrt(k*jlldens)
    if debug:
        print("rp4lp:maxflow: projecting from {0:d} to {1:d} rows".format(n-2,k))
    
    # generate projected matrix
    if debug:
        print("rp4lp:maxflow: computing projected matrix")
    if sparseFlag:
        TA = scipy.sparse.csr_matrix.dot(T,A) #sparse
    else:
        TA = np.dot(T,A)                     #dense

    tsample = time.time() - torg - t0
        
    ## solve projected formulation
    if debug:
        print("rp4lp:maxflow: solving projected instance")
    ## out projected .dat file
    # if debug:
    #    print("rp4lp:maxflow: writing projected .dat file")    
    # outProjDatMaxFlow(n,s,t,u,c,TA, AA,arc2col)
    # maxflowprj = AMPL()
    # maxflowprj.setOption("solver", LPSolverPrj)
    # solver_options = LPSolverPrj + "_options"
    # maxflowprj.setOption(solver_options, prjoptions)
    # maxflowprj.read(maxflowProjMod)
    # maxflowprj.readData(maxflowProjDat)
    # maxflowprj.solve()
    # # get optimal objective value
    # objfunprj = maxflowprj.getObjective("sourceflow")
    # fproj = objfunprj.value()
    # # get optimal solution
    # xvarprj = maxflowprj.getVariable("x")
    # xproj = np.zeros(xlen)
    # for h in range(xlen):
    #     xproj[h] = xvarprj[h+1].value()

    # solve projected problem with linprog (highs)
    Tb = np.zeros(k)
    l1 = np.zeros(xlen)
    u1 = np.array([u[col2arc[i]] for i in range(xlen)]) # UB on capacities
    LUbnd = list(zip(l1,u1))
    res = scipy.optimize.linprog(-c, A_eq=TA, b_eq=Tb, bounds=LUbnd, method="highs-ipm", options={"disp":True})
    xproj = res.get('x')
    fproj = np.dot(c,xproj)
        
    tprjslv = time.time() - tsample - torg - t0
        
    # solution retrieval: find xretr s.t. A*xretr = b-A*xproj
    ## method 1: pseudoinverse: xretr = pinv(A) * (b-A*xproj)
    # pinvA = np.linalg.pinv(A.todense()) 
    # rhs = A@xproj # because b=0 in maxflow problem
    # xretr = np.array(xproj - pinvA@rhs)
    # xretr = np.squeeze(xretr) # can't squeeze in-place
    ## method 2: least squares
    if sparseFlag:
        Axproj = A.dot(xproj)
        (xpart,status,iterations,err1,err2,normA,condA,normxx) = scipy.sparse.linalg.lsmr(A,Axproj) # for maxflow: b=0 so b-Axproj = -Axproj ('-' applied later)
        #if debug:
        #    print("rp4lp:maxflow:lsmr:status={0:d}".format(status))
        xretr = xproj - xpart  # for maxflow: '-' because b=0
    else:
        #xretr = xproj - np.dot(np.linalg.pinv(A),np.dot(A,xproj)) #slow
        xretr = xproj - np.linalg.lstsq(A, np.dot(A, xproj))      #faster
    #endif sparseFlag
    fretr = np.dot(c, xretr)
    
    if AltProjRetr:
        b = np.zeros(n-2)
        u1 = np.array([u[col2arc[i]] for i in range(xlen)]) # UB on capacities
        xretr,fretr,apstat,apitn,bnderr,feaserr = altProjRetrieval(xretr, A, b, c, fproj, objfundir, upp=u1, xp=xproj)
        if debug:
            print("post-apm:<c,xproj>={}, <c,xretr>={}".format(np.dot(c,xproj),fretr))
        
    tretr = time.time() - tprjslv - tsample - torg - t0    
    tprj = time.time() - torg - t0

    # performance measures
    if debug:
        print("rp4lp:maxflow: retrieving performance measures")    
    dimorg = n-2
    dimprj = k
    if sparseFlag:
        nnzorg = A.nnz
        nnzprj = TA.nnz
    else:
        nnzorg = np.count_nonzero(A)
        nnzprj = np.count_nonzero(TA)
    #endif sparseFlag
    normxstar = np.linalg.norm(xstar)
    normxproj = np.linalg.norm(xproj)
    normxretr = np.linalg.norm(xretr)
    dist_star_proj = np.linalg.norm(np.subtract(xstar, xproj))
    dist_star_retr = np.linalg.norm(np.subtract(xstar, xretr))
    dist_proj_retr = np.linalg.norm(np.subtract(xproj, xretr))
    avgAbinfeas = sum(abs(A[i].dot(xretr)) for i in range(n-2)) / (n-2)
    maxAbinfeas = max(abs(A[i].dot(xretr)) for i in range(n-2))
    if not isinstance(avgAbinfeas, float):
        avgAbinfeas = avgAbinfeas[0]
    if not isinstance(maxAbinfeas, float):
        maxAbinfeas = maxAbinfeas[0]
    avgineqinfeas = (sum([abs(min(xretr[i],0)) for i in range(xlen)]) + sum([max(xretr[i]-u[col2arc[i]],0) for i in range(xlen)])) / xlen
    maxineqinfeas = max(max([abs(min(xretr[i],0)) for i in range(xlen)]), max([max(xretr[i]-u[col2arc[i]],0) for i in range(xlen)]))
    print("rp4lp:out:Ax=b: A=({0:d},{1:d}),{2:d} TA=({3:d},{4:d}),{5:d}".format(dimorg,xlen,nnzorg,dimprj,xlen,nnzprj))
    print("rp4lp:out:norm: ||xstar||={0:f}".format(normxstar))
    print("rp4lp:out:norm: ||xproj||={0:f}".format(normxproj))
    print("rp4lp:out:norm: ||xretr||={0:f}".format(normxretr))
    print("rp4lp:out:dist: ||xstar-xproj||={0:f}".format(dist_star_proj))
    print("rp4lp:out:dist: ||xstar-xretr||={0:f}".format(dist_star_retr))
    print("rp4lp:out:dist: ||xproj-xretr||={0:f}".format(dist_proj_retr))    
    print("rp4lp:out:xretr:ineqerr: avg={0:f} max={1:f}".format(avgineqinfeas, maxineqinfeas))
    print("rp4lp:out:xretr:Aberr: avg={0:f} max={1:f}".format(avgAbinfeas, maxAbinfeas))
    print("rp4lp:out:objfun: fstar={0:f} fproj={1:f} fretr={2:f}".format(fstar,fproj,fretr))
    print("rp4lp:out:time:labels:torg,tprj,tsample,tprjslv,tretr")
    print("rp4lp:out:time:{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(torg,tprj,tsample,tprjslv,tretr))

    formulation = (dimorg, dimprj, xlen, constrdens, nnzorg, nnzprj, thetaest)
    objective = (fstar, fproj, fretr)
    solution = (xstar, xproj, xretr, normxstar, normxproj, normxretr, dist_star_proj, dist_star_retr, dist_proj_retr, normdual, theta)
    error = (avgineqinfeas, maxineqinfeas, avgAbinfeas, maxAbinfeas)
    cpu = (torg, tprj)
    return formulation, objfundir, objective, solution, error, cpu

def Diet(data, t0, jlleps, runOrg, fstar=None, xstar=None):

    if not solveOriginal:
        runOrg = False
    
    objfundir = -1.0 # minimization

    if not runOrg:
        assert(fstar is not None)
        assert(xstar is not None)
    
    # diet problem
    (m,n,bdict,cdict,Ddict) = parseDiet(data)
    
    if not runOrg:
        fstar = myInf
        xstar = np.ones(n)
        #assert(fstar is not None)
        #assert(xstar is not None)
    
    Dmat = np.zeros((m,n))
    for (i,j),v in Ddict.items():
        Dmat[i-offset,j-offset] = v
    b = np.zeros(m)
    for i in bdict:
        b[i-offset] = bdict[i]
    c = np.zeros(n)
    for j in cdict:
        c[j-offset] = cdict[j]
    
    if runOrg:
        # out original instance
        outDatDiet(m,n,bdict,cdict,Ddict)
        # solve original formulation
        diet = AMPL()
        diet.setOption("solver", LPSolverOrg)
        # for default settings, comment out the two following lines
        solver_options = LPSolverOrg + "_options"
        diet.setOption(solver_options, orgoptions)
        diet.read(dietMod)
        diet.readData(dietDat)
        diet.solve()
        # get optimal objective value
        objfun = diet.getObjective("cost")
        fstar = objfun.value()
        # get optimal solution
        xvar = diet.getVariable("x")
        xstar = np.zeros(n)
        for i in range(n):
            xstar[i] = xvar[i+offset].value()
        # dual solution norm
        cons = diet.getConstraint("nutrient")
        dual = np.zeros(m)
        for i in range(m):
            dual[i] = cons[i+offset].dual()
        normdual = np.linalg.norm(dual)
        # theta estimate
        qhat = np.zeros(n)
        eta = {i+offset:0 for i in range(m)}
        for (i,j) in Ddict:
            eta[i] += 1
        betaD = {i+offset:0 for i in range(m)}
        for j in range(n):
            j1 = j+offset
            for i in range(m):
                i1 = i + offset
                if (i1,j1) in Ddict and abs(Ddict[(i1,j1)]) > myZero:
                    beD = bdict[i1] / (eta[i1]*Ddict[(i1,j1)])
                    if beD > betaD[i1]:
                        betaD[i1] = beD
            qhat[j] = betaD[i1]            
        rhat = Dmat.dot(qhat) - b
        thetaest = sum(qhat) + sum(rhat)
        rstar = Dmat.dot(xstar) - b
        theta = sum(xstar) + sum(rstar)
        
    else:
        rstar = Dmat.dot(xstar) - b
        thetaest = sum(xstar) + sum(rstar)
        theta = thetaest
        normdual = 0.0

    torg = time.time() - t0

    ## projected formulation
    
    # density
    dD = len(Ddict) / (m*n)
    constrdens = (dD*n + 1) / (n+m)

    # sample random projector
    jlldens = RPDensFactor * constrdens
    if debug:
        print("rp4lp:diet: sampling RP with density={0:f}, |X|={1:d}".format(jlldens, n+m))
    k = int(round((1/jlleps)**2) * math.log(float(n+m)))
    if k > m:
        print("rp4lp:diet: {0:d} = k > m = {1:d}".format(k,m))
    T = sparse_gaussian(k, m, jlldens) / math.sqrt(k*jlldens)
    if debug:
        print("rp4lp:diet: projecting from {0:d} to {1:d} rows".format(m,k))
    
    # projected formulation type?
    dietRowsDiff = m # default: min cq + 1r
    dprjfrm2 = False
    if dietProjMod == "dietprj2.mod":
        # min cq + 1(r+ - r-)
        dprjfrm2 = True
        dietRowsDiff = 2*k
    
    # generate projected data
    Ds = scipy.sparse.csr_matrix(Dmat) # sparse version of D constraint matrix
    if dprjfrm2:
        # min cq + 1(r+ + r-) # vars r+,r- are in R^k
        # prepare A, I_k: then use (TA,I_k,-I_k)
        Is = scipy.sparse.identity(dietRowsDiff/2, format='csr') # I_k
        A = Ds
        TA = scipy.sparse.hstack([T@A, Is, -1.0*Is])
    else:        
        # default: min cq + 1r # vars r are in R^m
        # prepare A=(D,-I_m): then use TA=(TD,-TI_m)
        Is = scipy.sparse.identity(dietRowsDiff, format='csr') # I_m
        A = scipy.sparse.hstack([Ds, -1.0*Is])
        TA = scipy.sparse.csr_matrix.dot(T,A)             
    Tb = T.dot(b)

    tsample = time.time() - torg - t0

    ## solve projected formulation
    if debug:
        print("rp4lp:diet: solving projected instance")
    # output projected instance
    #outProjDatDiet(m,n,cdict,k,TA,Tb)
    # dietprj = AMPL()
    # dietprj.setOption("solver", LPSolverPrj)
    # solver_options = LPSolverPrj + "_options"
    # dietprj.setOption(solver_options, prjoptions)
    # dietprj.read(dietProjMod)
    # dietprj.readData(dietProjDat)
    # dietprj.solve()
    # # get optimal objective value
    # objfunprj = dietprj.getObjective("costprj")
    # fproj = objfunprj.value()
    # # get optimal solution
    # xsvarprj = dietprj.getVariable("xs")
    # xsproj = np.zeros(n+dietRowsDiff)
    # for j in range(n+dietRowsDiff):
    #     xsproj[j] = xsvarprj[j+1].value()
    # xproj = xsproj[0:n]

    # solve projected problem with linprog (highs)
    if dprjfrm2:
        cc = np.pad(c, (0,2*k), constant_values=2*n) # cc:(n+2k)-array pad w/ 'n'
    else:
        cc = np.pad(c, (0,m), constant_values=2*n)   # cd: (n+m)-array pad w/ 'n'
    res = scipy.optimize.linprog(cc, A_eq=TA, b_eq=Tb, method="highs-ipm", options={"disp":True})
    xsproj = res.get('x')
    xproj = xsproj[0:n]
    print(np.linalg.norm(c), np.linalg.norm(xproj))
    fproj = np.dot(c,xproj)

    tprjslv = time.time() - tsample - torg - t0
    
    if retrJLLMOR:
        ## this code snippet does not work with dprjfrm2=True
        # solution retrieval as in JLLMOR
        xstatus = {}
        xbasis = []
        for j in range(n+dietRowsDiff):
            xstatus[j] = xsvarprj[j+1].sstatus()
            if xstatus[j] == "bas":
                xbasis.append(j)
        AH = A.toarray()[:,xbasis]
        #xretrH = scipy.sparse.linalg.spsolve((AH.T).dot(AH), (AH.T).dot(b))
        #xretrH = np.linalg.solve((AH.T).dot(AH), (AH.T).dot(b))
        xretrH = np.linalg.lstsq(AH, b)[0]
        xsretr = np.zeros(n+dietRowsDiff)
        for i,j in enumerate(xbasis):
            xsretr[j] = xretrH[i]
    else:
        # solution retrieval as in JLLSDP: find xretr : A*xretr = b-A*xproj
        #   ie xretr = xproj + pinv(A)*(b-A*xproj)
        if dprjfrm2:
            # |x|=n
            Axproj = A.dot(xproj)
            xpart = scipy.sparse.linalg.lsmr(A,b-Axproj)[0]
            xsretr = xproj + xpart
            cc = c
            xsproj = xproj
        else:
            # |x|=n+m
            Axsproj = A.dot(xsproj)
            xspart = scipy.sparse.linalg.lsmr(A,b-Axsproj)[0]
            xsretr = xsproj + xspart

    # APM
    if AltProjRetr:
        xsretr,fretr,apstat,apitn,bnderr,feaserr = altProjRetrieval(xsretr, A, b, cc, fproj, objfundir, xp=xsproj)
    xretr = xsretr[0:n]
        
    # obj fun @ xretr    
    fretr = np.dot(c[0:n], xretr)
    # CPU time
    tretr = time.time() - tprjslv - tsample - torg - t0    
    tprj = time.time() - torg - t0

    # performance measures
    dimorg = m
    dimprj = k
    nnzorg = A.nnz
    nnzprj = TA.nnz
    normxstar = np.linalg.norm(xstar)
    normxproj = np.linalg.norm(xproj)
    normxretr = np.linalg.norm(xretr)
    dist_star_proj = np.linalg.norm(np.subtract(xstar, xproj))
    dist_star_retr = np.linalg.norm(np.subtract(xstar, xretr))
    dist_proj_retr = np.linalg.norm(np.subtract(xproj, xretr))
    avgineqinfeas = sum([abs(min(xretr[i],0)) for i in range(n)]) / n
    maxineqinfeas = max([abs(min(xretr[i],0)) for i in range(n)])
    Dx = Dmat.dot(xretr)
    avgAbinfeas = sum(abs(min(Dx[i]-b[i],0)) for i in range(m)) / m
    maxAbinfeas = max(abs(min(Dx[i]-b[i],0)) for i in range(m))
    print("rp4lp:out:Ax=b: A=({0:d},{1:d}),{2:d} TA=({3:d},{4:d}),{5:d}".format(dimorg,n+m,nnzorg,dimprj,n+m,nnzprj))
    print("rp4lp:out:norm: ||xstar||={0:f}".format(normxstar))
    print("rp4lp:out:norm: ||xproj||={0:f}".format(normxproj))
    print("rp4lp:out:norm: ||xretr||={0:f}".format(normxretr))
    print("rp4lp:out:dist: ||xstar-xproj||={0:f}".format(dist_star_proj))
    print("rp4lp:out:dist: ||xstar-xretr||={0:f}".format(dist_star_retr))
    print("rp4lp:out:dist: ||xproj-xretr||={0:f}".format(dist_proj_retr))    
    print("rp4lp:out:xretr:ineqerr: avg={0:f} max={1:f}".format(avgineqinfeas, maxineqinfeas))
    print("rp4lp:out:xretr:Aberr: avg={0:f} max={1:f}".format(avgAbinfeas, maxAbinfeas))
    print("rp4lp:out:objfun: fstar={0:f} fproj={1:f} fretr={2:f}".format(fstar,fproj,fretr))
    print("rp4lp:out:time:labels:torg,tprj,tsample,tprjslv,tretr")
    print("rp4lp:out:time:{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(torg,tprj,tsample,tprjslv,tretr))
    formulation = (dimorg, dimprj, n, constrdens, nnzorg, nnzprj, thetaest)
    objective = (fstar, fproj, fretr)
    solution = (xstar, xproj, xretr, normxstar, normxproj, normxretr, dist_star_proj, dist_star_retr, dist_proj_retr, normdual, theta)
    error = (avgineqinfeas, maxineqinfeas, avgAbinfeas, maxAbinfeas)
    cpu = (torg, tprj)
    return formulation, objfundir, objective, solution, error, cpu

def QuantReg(data, t0, jlleps, runOrg, fstar=None, xstar=None):

    objfundir = -1.0 # minimization

    if not solveOriginal:
        runOrg = False    
    
    if not runOrg:
        assert(fstar is not None)
        assert(xstar is not None)

    # quantile regression
    (m,p,bidx,tau,Ddict) = parseQuantReg(data)

    # encode matrices
    D = np.zeros((m,p-1))
    Dnnz = 0
    b = np.zeros(m)
    for (i,j),v in Ddict.items():
        if j == bidx:
            b[i-offset] = v
        else:
            if j < bidx:
                h = j
            else:
                h = j-1
            D[i-offset,h-offset] = v
            Dnnz += 1
    # work out density
    dD = Dnnz / (m*(p-1))
    constrdens = (dD*p + 2.0) / (p + 2*m)
        
    if runOrg:
        # write original instance
        outDatQuantReg(m,p,bidx,tau,Ddict)
        # solve original formulation
        quantreg = AMPL()
        quantreg.setOption("solver", LPSolverOrg)
        # for default settings, comment out the two following lines
        solver_options = LPSolverOrg + "_options"
        quantreg.setOption(solver_options, orgoptions)
        quantreg.read(quantregMod)
        quantreg.readData(quantregDat)
        quantreg.solve()
        # get optimal objective value
        objfun = quantreg.getObjective("error")
        fstar = objfun.value()
        # get optimal solution
        varbeta = quantreg.getVariable("beta")
        betastar = np.zeros(p-1)
        h = 0
        for j in range(p):
            if j+offset != bidx:
                betastar[h] = varbeta[j+offset].value()
                h += 1
        varup = quantreg.getVariable("up")
        upstar = np.zeros(m)
        for i in range(m):
            upstar[i] = varup[i+1].value()
        varum = quantreg.getVariable("um")
        umstar = np.zeros(m)
        for i in range(m):
            umstar[i] = varum[i+1].value()
        xstar = np.concatenate((betastar, upstar, umstar))
        # dual solution norm and theta estimate
        cons = quantreg.getConstraint("quantile")
        dual = np.zeros(m)
        for i in range(m):
            dual[i] = cons[i+offset].dual()
        normdual = np.linalg.norm(dual)

    else: # not runOrg
        normdual = 0
        betastar = xstar[0:p-1]
        upstar = xstar[p-1:p-1+m]
        umstar = xstar[p-1+m:p-1+2*m]

    thetaest = p + sum([abs(b[i]) for i in range(m)])
    theta = sum(betastar) + sum(upstar) + sum(umstar)
    torg = time.time() - t0

    # create matrix A for Ax=b constraints
    Ds = scipy.sparse.csr_matrix(D)
    Is = scipy.sparse.identity(m, format='csr')
    A = scipy.sparse.hstack([Ds, Is, -Is])
    cols = p-1+2*m
    
    # sample random projector
    jlldens = RPDensFactor * constrdens
    if debug:
        print("rp4lp:quantreg: sampling RP with density={0:f} |X|={1:d}".format(jlldens, cols))        
    k = int(round((1/jlleps)**2) * math.log(float(cols)))
    if k > m:
        print("rp4lp:quantreg: {0:d} = k > m = {1:d}".format(k,m))
    T = sparse_gaussian(k, m, jlldens) / math.sqrt(k*jlldens)
    if debug:
        print("rp4lp:quantreg: projecting from {0:d} to {1:d} rows".format(m,k))
    
    # generate projected data
    TA = scipy.sparse.csr_matrix.dot(T,A)
    Tb = T.dot(b)

    tsample = time.time() - torg - t0    

    ## solve projected formulation
    if debug:
        print("rp4lp:quantreg: solving projected instance")
    ## output projected instance
    #outProjDatQuantReg(m,p,tau,k,TA,Tb)
    # quantregprj = AMPL()
    # quantregprj.setOption("solver", LPSolverPrj)
    # solver_options = LPSolverPrj + "_options"
    # quantregprj.setOption(solver_options, prjoptions)
    # quantregprj.read(quantregProjMod)
    # quantregprj.readData(quantregProjDat)
    # quantregprj.solve()
    # # get optimal objective value
    # objfunprj = quantregprj.getObjective("error")
    # fproj = objfunprj.value()
    # # get optimal solution
    # varbetaproj = quantregprj.getVariable("beta")
    # betaproj = np.zeros(p-1)
    # h = 0
    # for j in range(p):
    #     if j+offset != bidx:
    #         betaproj[h] = varbetaproj[j+offset].value()
    #         h += 1
    # varupproj = quantregprj.getVariable("up")
    # upproj = np.zeros(m)
    # for i in range(m):
    #     upproj[i] = varupproj[i+1].value()
    # varumproj = quantregprj.getVariable("um")
    # umproj = np.zeros(m)
    # for i in range(m):
    #     umproj[i] = varumproj[i+1].value()
    # xproj = np.concatenate((betaproj, upproj, umproj))

    c = np.concatenate([np.zeros(p-1), tau*np.ones(m), (1-tau)*np.ones(m)])
    lb = np.concatenate([-myInf*np.ones(p-1), np.zeros(2*m)])
    ub = myInf*np.ones(p-1+2*m)
    LU = list(zip(lb,ub))
    res = scipy.optimize.linprog(c, A_eq=TA, b_eq=Tb, bounds=LU, method="highs-ipm", options={"disp":True, "ipm_optimality_tolerance":1e-5})
    xproj = res.get('x')
    fproj = np.dot(c,xproj)
    
    tprjslv = time.time() - tsample - torg - t0    
    
    # solution retrieval as in JLLSDP: find xretr : A*xretr = b-A*xproj
    #   ie xretr = xproj + pinv(A)*(b-A*xproj)
    Axproj = A.dot(xproj)
    (xpart,status,iterations,err1,err2,normA,condA,normxx) = scipy.sparse.linalg.lsmr(A,b-Axproj)
    #if debug:
    #    print("rp4lp:quantreg:lsmr:status={0:d}".format(status))
    xretr = xproj + xpart

    if AltProjRetr:
        xretr,fretr,apstat,apitn,bnderr,feaserr = altProjRetrieval(xretr, A, b, c, fproj, objfundir, xp=xproj)
    
    betaretr = xretr[0:p-1]
    upretr = xretr[p-1:p+m-1]
    umretr = xretr[p+m-1:p+2*m-1]
    fretr = tau*sum(upretr) + (1-tau)*sum(umretr)
    tretr = time.time() - tprjslv - tsample - torg - t0    
    tprj = time.time() - torg - t0

    # performance measures
    dimorg = m
    dimprj = k
    nnzorg = A.nnz
    nnzprj = TA.nnz
    normxstar = np.linalg.norm(xstar)
    normxproj = np.linalg.norm(xproj)
    normxretr = np.linalg.norm(xretr)
    dist_star_proj = np.linalg.norm(np.subtract(xstar, xproj))
    dist_star_retr = np.linalg.norm(np.subtract(xstar, xretr))
    dist_proj_retr = np.linalg.norm(np.subtract(xproj, xretr))
    avgineqinfeas = sum([abs(min(upretr[i],0)) for i in range(m)]) / m
    avgineqinfeas += sum([abs(min(umretr[i],0)) for i in range(m)]) / m
    maxineqinfeas = max([abs(min(upretr[i],0)) for i in range(m)])
    maxineqinfeas += max([abs(min(umretr[i],0)) for i in range(m)])
    Ax = A.dot(xretr)
    avgAbinfeas = sum(abs(Ax[i]-b[i]) for i in range(m)) / m
    maxAbinfeas = max(abs(Ax[i]-b[i]) for i in range(m))
    print("rp4lp:out:Ax=b: A=({0:d},{1:d}),{2:d} TA=({3:d},{4:d}),{5:d}".format(dimorg,cols,nnzorg,dimprj,cols,nnzprj))
    print("rp4lp:out:norm: ||xstar||={0:f}".format(normxstar))
    print("rp4lp:out:norm: ||xproj||={0:f}".format(normxproj))
    print("rp4lp:out:norm: ||xretr||={0:f}".format(normxretr))
    print("rp4lp:out:dist: ||xstar-xproj||={0:f}".format(dist_star_proj))
    print("rp4lp:out:dist: ||xstar-xretr||={0:f}".format(dist_star_retr))
    print("rp4lp:out:dist: ||xproj-xretr||={0:f}".format(dist_proj_retr))    
    print("rp4lp:out:xretr:ineqerr: avg={0:f} max={1:f}".format(avgineqinfeas, maxineqinfeas))
    print("rp4lp:out:xretr:Aberr: avg={0:f} max={1:f}".format(avgAbinfeas, maxAbinfeas))
    print("rp4lp:out:objfun: fstar={0:f} fproj={1:f} fretr={2:f}".format(fstar,fproj,fretr))
    print("rp4lp:out:time:labels:torg,tprj,tsample,tprjslv,tretr")
    print("rp4lp:out:time:{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(torg,tprj,tsample,tprjslv,tretr))
    formulation = (dimorg, dimprj, p, constrdens, nnzorg, nnzprj, thetaest)
    objective = (fstar, fproj, fretr)
    solution = (xstar, xproj, xretr, normxstar, normxproj, normxretr, dist_star_proj, dist_star_retr, dist_proj_retr, normdual, theta)
    error = (avgineqinfeas, maxineqinfeas, avgAbinfeas, maxAbinfeas)
    cpu = (torg, tprj)
    return formulation, objfundir, objective, solution, error, cpu
    
def BasisPursuit(data, t0, jlleps, runOrg, fstar=None, xstar=None):

    objfundir = -1.0 # minimization

    if not solveOriginal:
        runOrg = False
    
    if not runOrg:
        assert(fstar is not None)
        assert(xstar is not None)

    # basis pursuit
    (m,n,s,org,enc,Adict) = parseBasisPursuit(data)

    if runOrg:    
        # output original instance
        outDatBasisPursuit(m,n,s,org,enc,Adict)
        # solve original formulation
        basispursuit = AMPL()
        basispursuit.setOption("solver", LPSolverOrg)
        # for default settings, comment out the two following lines
        solver_options = LPSolverOrg + "_options"
        basispursuit.setOption(solver_options, orgoptions)
        basispursuit.read(basispursuitMod)
        basispursuit.readData(basispursuitDat)
        basispursuit.solve()
        # get optimal objective value
        objfun = basispursuit.getObjective("ell1error")
        fstar = objfun.value()
        # get optimal solution
        xvar = basispursuit.getVariable("x")
        xstar = np.zeros(n)
        for i in range(n):
            xstar[i] = xvar[i+offset].value()
        # dual solution norm and theta estimate
        cons = basispursuit.getConstraint("decoding")
        dual = np.zeros(m)
        for i in range(m):
            dual[i] = cons[i+offset].dual()
        normdual = np.linalg.norm(dual)
    else:
        normdual = 0
        
    thetaest = 2*n*n
    theta = sum(xstar)
    torg = time.time() - t0
    
    # matrices (A,b)
    A = np.zeros((m,n))
    for (i,j),v in Adict.items():
        A[i-offset,j-offset] = v
    b = np.zeros(m)
    for i,v in enc.items():
        b[i-offset] = v

    # density of A
    nnzorg = np.count_nonzero(A)
    constrdens = nnzorg / (m*n)
    
    # sample random projector
    jlldens = RPDensFactor * constrdens
    if debug:
        print("rp4lp:basispursuit: sampling RP with density={0:f} |X|={1:d}".format(jlldens, n))
    k = int(round((1/jlleps)**2) * math.log(float(n)))
    if k > m:
        print("rp4lp:basispursuit: {0:d} = k > m = {1:d}".format(k,m))
    T = sparse_gaussian(k, m, jlldens) / math.sqrt(k*jlldens)
    if debug:
        print("rp4lp:basispursuit: projecting from {0:d} to {1:d} rows".format(m,k))
 
    # generate projected data
    TA = scipy.sparse.csr_matrix.dot(T,A) 
    Tb = T.dot(b)
    
    tsample = time.time() - torg - t0           
        
    ## solve projected formulation
    if debug:
        print("rp4lp:basispursuit: solving projected instance")
    # output projected instance
    outProjDatBasisPursuit(k,n,s,org, TA,Tb)
    basispursuitprj = AMPL()
    #basispursuitprj.setOption("solver", LPSolverPrj)
    basispursuitprj.setOption("solver", "cplex") # force cplex
    #solver_options = LPSolverPrj + "_options"
    #solver_options = "cplex_options"             # force cplex options
    ## baropt is much slower than simplex on basis pursuit - use default opts
    #basispursuitprj.setOption(solver_options, prjcplexoptions)
    basispursuitprj.read(basispursuitProjMod)
    basispursuitprj.readData(basispursuitProjDat)
    basispursuitprj.solve()    
    # get optimal objective value
    objfunprj = basispursuitprj.getObjective("ell1error")
    fproj = objfunprj.value()
    # get optimal solution
    xvarprj = basispursuitprj.getVariable("x")
    xproj = np.zeros(n)
    for j in range(n):
        xproj[j] = xvarprj[j+1].value()
    svvarprj = basispursuitprj.getVariable("sv")
    svproj = np.zeros(n)
    for j in range(n):
        svproj[j] = svvarprj[j+1].value()

    ## highs is inaccurate on basis pursuit
    # c = np.concatenate([np.zeros(n), np.ones(n)])
    # Aup = np.hstack([np.identity(n), -1.0*np.identity(n)])
    # Adn = np.hstack([-1.0*np.identity(n), -1.0*np.identity(n)])
    # Aineq = np.vstack([Aup, Adn])
    # bineq = np.zeros(2*n)
    # TA = np.hstack([TA, np.zeros((k,n))])
    # res = scipy.optimize.linprog(c, A_eq=TA, b_eq=Tb, A_ub=Aineq, b_ub=bineq, method="highs-ipm", options={"disp":True, 'ipm_optimality_tolerance':1e-5})
    # xsproj = res.get('x') # [0:n]:signal; [n:2*n]:absvalreform
    # fproj = np.dot(c,xsproj)
    # xproj = xsproj[0:n]
    
    tprjslv = time.time() - tsample - torg - t0    
        
    # solution retrieval as in JLLSDP: find xretr : A*xretr = b-A*xproj
    #   ie xretr = xproj + pinv(A)*(b-A*xproj)
    Axproj = A.dot(xproj)
    bmAxproj = b - Axproj
    (xpart, rowerror, rank, s) = np.linalg.lstsq(A, bmAxproj, rcond=None)
    error = sum(rowerror)
    if debug:
        print("rp4lp:basispursuit:lstsq: error={0:f}".format(error))    
    xretr = xproj + xpart

    # no bound constrs on x in basis pursuit => no need for altProjRetrieval
    
    fretr = sum([abs(xretr[j]) for j in range(n)])
    tretr = time.time() - tprjslv - tsample - torg - t0    
    tprj = time.time() - torg - t0    
        
    # performance measures
    dimorg = m
    dimprj = k
    nnzprj = np.count_nonzero(TA)
    normxstar = np.linalg.norm(xstar)
    normxproj = np.linalg.norm(xproj)
    normxretr = np.linalg.norm(xretr)
    dist_star_proj = np.linalg.norm(np.subtract(xstar, xproj))
    dist_star_retr = np.linalg.norm(np.subtract(xstar, xretr))
    dist_proj_retr = np.linalg.norm(np.subtract(xproj, xretr))
    avgineqinfeas = 0 # no inequalities x >= 0
    maxineqinfeas = 0 # no inequalities x >= 0
    Ax = A.dot(xretr)
    avgAbinfeas = sum(abs(Ax[i]-b[i]) for i in range(m)) / m
    maxAbinfeas = max(abs(Ax[i]-b[i]) for i in range(m))
    print("rp4lp:out:Ax=b: A=({0:d},{1:d}),{2:d} TA=({3:d},{4:d}),{5:d}".format(dimorg,n,nnzorg,dimprj,n,nnzprj))
    print("rp4lp:out:norm: ||xstar||={0:f}".format(normxstar))
    print("rp4lp:out:norm: ||xproj||={0:f}".format(normxproj))
    print("rp4lp:out:norm: ||xretr||={0:f}".format(normxretr))
    print("rp4lp:out:dist: ||xstar-xproj||={0:f}".format(dist_star_proj))
    print("rp4lp:out:dist: ||xstar-xretr||={0:f}".format(dist_star_retr))
    print("rp4lp:out:dist: ||xproj-xretr||={0:f}".format(dist_proj_retr))    
    print("rp4lp:out:xretr:ineqerr: avg={0:f} max={1:f}".format(avgineqinfeas, maxineqinfeas))
    print("rp4lp:out:xretr:Aberr: avg={0:f} max={1:f}".format(avgAbinfeas, maxAbinfeas))
    print("rp4lp:out:objfun: fstar={0:f} fproj={1:f} fretr={2:f}".format(fstar,fproj,fretr))
    print("rp4lp:out:time:labels:torg,tprj,tsample,tprjslv,tretr")
    print("rp4lp:out:time:{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(torg,tprj,tsample,tprjslv,tretr))
    formulation = (dimorg, dimprj, n, constrdens, nnzorg, nnzprj, thetaest)
    objective = (fstar, fproj, fretr)
    solution = (xstar, xproj, xretr, normxstar, normxproj, normxretr, dist_star_proj, dist_star_retr, dist_proj_retr, normdual, theta)
    error = (avgineqinfeas, maxineqinfeas, avgAbinfeas, maxAbinfeas)
    cpu = (torg, tprj)
    return formulation, objfundir, objective, solution, error, cpu

def Uniform(data, t0, jlleps, runOrg, fstar=None, xstar=None):

    objfundir = -1.0 # minimization
    
    if not solveOriginal:
        runOrg = False

    if not runOrg:
        assert(fstar is not None)
        assert(xstar is not None)

    # basis pursuit
    (m,n,bdict,cdict,Adict) = parseUniform(data)

    if runOrg:    
        # output original instance
        outDatUniform(m,n,bdict,cdict,Adict)

        # solve original formulation
        uniform = AMPL()
        uniform.setOption("solver", LPSolverOrg)
        # for default settings, comment out the two following lines
        solver_options = LPSolverOrg + "_options"
        uniform.setOption(solver_options, orgoptions)
        uniform.read(uniformMod)
        uniform.readData(uniformDat)
        uniform.solve()

        # get optimal objective value
        objfun = uniform.getObjective("obj")
        fstar = objfun.value()
        
        # get optimal solution
        xvar = uniform.getVariable("x")
        xstar = np.zeros(n)
        for i in range(n):
            xstar[i] = xvar[i+offset].value()
        
        # dual solution norm and theta estimate
        cons = uniform.getConstraint("con")
        dual = np.zeros(m)
        for i in range(m):
            dual[i] = cons[i+offset].dual()
        normdual = np.linalg.norm(dual)
    else:
        normdual = 0
        
    thetaest = 2*n*n ## TODO: estimate this specifically for uniform
    theta = sum(xstar)

    torg = time.time() - t0
    
    # matrices (A,b)
    A = np.zeros((m,n))
    for (i,j),v in Adict.items():
        A[i-offset,j-offset] = v
    b = np.zeros(m)
    for i,v in bdict.items():
        b[i-offset] = v
    c = np.zeros(n)
    for i,v in cdict.items():
        c[i-offset] = v

    # density of A
    nnzorg = np.count_nonzero(A)
    constrdens = nnzorg / (m*n)

    # sample random projector
    jlldens = RPDensFactor * constrdens
    if debug:
        print("rp4lp:uniform: sampling RP with density={0:f} |X|={1:d}".format(jlldens, n))
    k = int(round((1/jlleps)**2) * math.log(float(n)))
    if k > m:
        print("rp4lp:uniform: {0:d} = k > m = {1:d}".format(k,m))
    T = sparse_gaussian(k, m, jlldens) / math.sqrt(k*jlldens)
    if debug:
        print("rp4lp:uniform: projecting from {0:d} to {1:d} rows".format(m,k))
    
    # generate projected data
    A = scipy.sparse.csr_matrix(A)
    TA = scipy.sparse.csr_matrix.dot(T,A) 
    Tb = T.dot(b)

    tsample = time.time() - torg - t0           
    
    # output projected instance
    outProjDatUniform(k,n,cdict,TA,Tb)

    # solve projected formulation
    if debug:
        print("rp4lp:uniform: solving projected instance")
    uniformprj = AMPL()
    uniformprj.setOption("solver", LPSolverPrj)
    solver_options = LPSolverPrj + "_options"
    uniformprj.setOption(solver_options, prjoptions)
    uniformprj.read(uniformProjMod)
    uniformprj.readData(uniformProjDat)
    uniformprj.solve()
    
    # get optimal objective value
    objfunprj = uniformprj.getObjective("obj")
    fproj = objfunprj.value()

    # get optimal solution
    xvarprj = uniformprj.getVariable("x")
    xproj = np.zeros(n)
    for j in range(n):
        xproj[j] = xvarprj[j+1].value()

    tprjslv = time.time() - tsample - torg - t0    
        
    # solution retrieval as in JLLSDP: find xretr : A*xretr = b-A*xproj
    #   ie xretr = xproj + pinv(A)*(b-A*xproj)
    Axproj = A.dot(xproj)
    (xpart,stat,itns,e1,e2,nA,cA,nx) = scipy.sparse.linalg.lsmr(A,b-Axproj)
    xretr = xproj + xpart

    if AltProjRetr:
        xretr,fretr,apstat,apitn,bnderr,feaserr = altProjRetrieval(xretr,A,b,c,fproj,objfundir, xp=xproj)
    fretr = np.dot(c,xretr)
    tretr = time.time() - tprjslv - tsample - torg - t0    
    tprj = time.time() - torg - t0
    
    # performance measures
    dimorg = m
    dimprj = k
    nnzorg = A.nnz
    nnzprj = TA.nnz
    normxstar = np.linalg.norm(xstar)
    normxproj = np.linalg.norm(xproj)
    normxretr = np.linalg.norm(xretr)
    dist_star_proj = np.linalg.norm(np.subtract(xstar, xproj))
    dist_star_retr = np.linalg.norm(np.subtract(xstar, xretr))
    dist_proj_retr = np.linalg.norm(np.subtract(xproj, xretr))
    avgineqinfeas = sum([abs(min(xretr[i],0)) for i in range(m)]) / m
    maxineqinfeas = max([abs(min(xretr[i],0)) for i in range(m)])
    Ax = A.dot(xretr)
    avgAbinfeas = sum(abs(Ax[i]-b[i]) for i in range(m)) / m
    maxAbinfeas = max(abs(Ax[i]-b[i]) for i in range(m))
    print("rp4lp:out:Ax=b: A=({0:d},{1:d}),{2:d} TA=({3:d},{4:d}),{5:d}".format(dimorg,m,nnzorg,dimprj,n,nnzprj))
    print("rp4lp:out:norm: ||xstar||={0:f}".format(normxstar))
    print("rp4lp:out:norm: ||xproj||={0:f}".format(normxproj))
    print("rp4lp:out:norm: ||xretr||={0:f}".format(normxretr))
    print("rp4lp:out:dist: ||xstar-xproj||={0:f}".format(dist_star_proj))
    print("rp4lp:out:dist: ||xstar-xretr||={0:f}".format(dist_star_retr))
    print("rp4lp:out:dist: ||xproj-xretr||={0:f}".format(dist_proj_retr))    
    print("rp4lp:out:xretr:ineqerr: avg={0:f} max={1:f}".format(avgineqinfeas, maxineqinfeas))
    print("rp4lp:out:xretr:Aberr: avg={0:f} max={1:f}".format(avgAbinfeas, maxAbinfeas))
    print("rp4lp:out:objfun: fstar={0:f} fproj={1:f} fretr={2:f}".format(fstar,fproj,fretr))
    print("rp4lp:out:time:labels:torg,tprj,tsample,tprjslv,tretr")
    print("rp4lp:out:time:{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(torg,tprj,tsample,tprjslv,tretr))
    formulation = (dimorg, dimprj, n, constrdens, nnzorg, nnzprj, thetaest)
    objective = (fstar, fproj, fretr)
    solution = (xstar, xproj, xretr, normxstar, normxproj, normxretr, dist_star_proj, dist_star_retr, dist_proj_retr, normdual, theta)
    error = (avgineqinfeas, maxineqinfeas, avgAbinfeas, maxAbinfeas)
    cpu = (torg, tprj)
    return formulation, objfundir, objective, solution, error, cpu

    
# estimate a fast dual bound for LP
def poorManDualBound(A,b,c,tol):
    # solve yA=c at min error (overdetermined)
    #   i.e. A'y=c'
    m = A.shape[0]
    n = A.shape[1]
    # is matrix sparse?
    spA = scipy.sparse.issparse(A)
    if spA:
        # sparse least squares
        (y,stat,itn,e1,e2,nA,cA,nx) = scipy.sparse.linalg.lsmr(A.T,c)
    else:
        # dense least squares
        y,rs,rk,sv = np.linalg.lstsq(A.T,c)
    # correct y's so that yA^j <= c_j for all j
    while True:
        s = A.T.dot(y) - c
        j = np.argmax(s)
        if s[j] > tol:
            if debug:
                print("rp4lp:dualbound: s[{:d}]={:f}>0".format(j,s[j]))
            Aj = A.getcol(j)
            for i in range(m):
                Aij = Aj[i,0]
                if abs(Aij) > myZero:
                    y[i] -= s[j] / (m * Aij)
        else:
            break    
    # compute dual bound yb
    dualbound = y.dot(b)
    return dualbound

# adapted to scipy.sparse
#   from https://mail.python.org/pipermail/scipy-user/2009-May/021308.html
#   doesn't converge
def project_subspace(z,A,b, eps=np.finfo(float).eps):
    """ Project a vector onto the subspace defined by "dot(A,x) = b".
    """
    m, n = A.shape
    #u, s, vh = np.linalg.svd(A)
    u, s, vh = scipy.sparse.linalg.svds(A, k=min(m,n)-1)
    # the three following magic lines from
    # Roger Hung's answer @
    #https://stackoverflow.com/questions/50884533/numpy-svd-vs-scipy-sparse-svds
    u = -u[:,::-1]
    vh = -vh[::-1,:]
    s = s[::-1]
    # Find the first singular value to drop below the cutoff.
    bad = (s < s[0] * eps)
    i = bad.searchsorted(1)
    # if i < m-1:
    #     rcond = s[i]
    # else:
    #     rcond = -1
    # sol = np.linalg.lstsq(A, b, rcond=rcond)
    sol = scipy.sparse.linalg.lsmr(A, b)
    x0 = sol[0]
    null_space = vh[i:]
    z_proj = x0 + (null_space * np.dot(null_space,z-x0)[:,np.newaxis]).sum(axis=0)
    return z_proj

# project on a subspace with Alg 10.31 in [Combettes & Pesquet 2011]
#   doesn't converge
def projectOnSubspace(xstart,A,b,tol=AltProjTol,maxitn=AltProjMaxItn):
    m, n = A.shape
    x = xstart
    z = {i:np.copy(x) for i in range(m)}
    p = {i:None for i in range(m)}
    A = np.array(A.todense())
    errstart = np.linalg.norm(A@x-b)
    for itn in range(maxitn):
        for i in range(m):
            #Ai = np.array(A.getrow(i).todense())
            #Ai = np.squeeze(Ai) # squeeze won't work in-place
            Ai = np.squeeze(A[i])
            p[i] = z[i] - (np.dot(Ai,z[i])-b[i]) * (Ai / np.linalg.norm(Ai))
        x = np.sum([p[i] for i in range(m)], axis=0) / m
        for i in range(m):
            z[i] = x + z[i] - p[i]
    errend = np.linalg.norm(A@x-b)
    print("rp4lp:projectOnSubspace: err0={} err1={}".format(errstart,errend))
    return x

# bisection search:
#  optimize in direction od the linear form <c,x> on segment [x0,x1]
#  using feasibility oracle Ax=b
def bisectionSearch(xL,xU,A,b,c,od,tol=AltProjTol):
    fL = np.dot(c,xL)
    fU = np.dot(c,xU)
    # make sure the optimization direction goes from xL to xU
    if (fL < fU and od < 0) or (fL > fU and od > 0):
        # swap 0 and 1
        ftmp = fL
        fL = fU
        fU = ftmp
        ftmp = fL
        fL = fU
        fU = ftmp
    # run bisection
    itn = 0
    xstar = np.copy(xL)
    fstar = fL
    while fU - fL > tol:
        x = (xL + xU) / 2
        fx = np.dot(c,x)
        if np.linalg.norm(A@x-b) < tol:
            # x is feasible, set xL=x
            xL = x
            fL = fx
            if od*fx > od*fstar:
                # local opt
                fstar = fx
                xstar = x
        else:
            # x is infeasible, set xU=x
            xU = x
            fU = fx
        print("rp4lp:bisection: itn={} f={:.3f} fL={:.3f} fU={:.3f}".format(itn, fx, fL, fU))
        itn += 1
    return xstar

def projObjFun(x,x0):
    return np.linalg.norm(np.subtract(x,x0))

# alternating projection method for better retrieved solutions (dykstra's APM)
#   called after standard pseudoinverse retrieval, so xstart feas wrt Ax=b
#   fproj,objfundir are not used in this implementation
#     (but see altProjRetrieval-2211.py in this dir)
def altProjRetrieval(xstart,A,b,c,fproj,objfundir,low=None,upp=None,tol=AltProjTol,maxitn=AltProjMaxItn, xp=None):
    aprcpu0 = time.time()
    m = A.shape[0]
    n = A.shape[1]
    # is matrix sparse?
    spA = scipy.sparse.issparse(A)
    # check l,u
    if low is None:
        low = np.zeros(n)
    if upp is None:
        upp = myInf*np.ones(n)
    # original objective cost with xstart
    forg = np.dot(c,xstart)
    # don't change xstart during APM
    x = np.copy(xstart) 
    xprev = np.copy(x)
    status = "1:maxitn"
    iterations = maxitn

    t0 = time.time()
    
    ## compute pseudoinverse of A once and for all
    #pinvA = scipy.linalg.pinv(A.todense())
    xcheck = scipy.sparse.linalg.lsmr(A,b)[0]

    ## go!
    if AltProjMethod == "Dykstra":
        # Dykstra's APM - since xstart feasible in Ax=b, 1st proj on [low,upp]
        p = np.zeros(n)
        q = np.zeros(n)
        y = np.zeros(n)
        for itn in range(maxitn):
            for j in range(n):
                xprev[j] = x[j]
            
            # verify bound error wrt low <= x <= upp
            bnderr = sum([abs(min(x[j] - low[j], 0)) for j in range(n)])
            bnderr+= sum([abs(min(upp[j] - x[j], 0)) for j in range(n)])
            if bnderr > tol:
                # bound error, project on low <= x+p <= upp, call proj y
                for j in range(n):
                    if x[j]+p[j] < low[j]:
                        y[j] = low[j]
                    elif x[j]+p[j] > upp[j]:
                        y[j] = upp[j]
                    else:
                        y[j] = x[j]+p[j]
            else: 
                # no bound error: exit
                status = "1:nobnderr"
                iterations = itn + 1
                x = xprev
                break
            # Dykstra: update p
            p = x+p - y
            # verify feasibility error wrt Ay=b (not y+q)
            feaserr = np.linalg.norm(A.dot(y)-b) 
            if feaserr > tol:
                # feasibility error, project on A(y+q)=b
                rhs = b + A.dot(y+q) 
                if spA:
                    ### sparse
                    ## method 1: pseudoinverse (slow!)
                    #x = (y+q) + pinvA@(b-A@(y+q))
                    ## method 2: call an optimization solver (even slower!)
                    #res = scipy.optimize.minimize(projObjFun,y+q,args=(y+q),bounds=None, tol=tol, constraints=(scipy.optimize.LinearConstraint(A,b,b)))
                    ## method 3: least squares - doesn't solve what we want
                    #x = scipy.sparse.linalg.lsmr(A,rhs,x0=y+q)[0]
                    ## method 4: project_subspace (taken from the internet)
                    #x = project_subspace(y+q, A, b)
                    ## method 5: projectOnSubspace (sub-APM)
                    #x = projectOnSubspace(y+q, A, b)
                    ## method 6: line search between xcheck and xbar
                    #xcheck = scipy.sparse.linalg.lsmr(A,b)[0] #COMPUTED ONCE
                    xhat = scipy.sparse.linalg.lsmr(A,rhs)[0]
                    xbar = xhat - (y+q)
                    alph = xcheck - xbar
                    bet = y+q - xbar
                    x=(np.dot(alph,bet)/(np.linalg.norm(alph)**2))*alph + xbar
                    ## method 7: lsqr
                    #x = scipy.sparse.linalg.lsqr(A,rhs,x0=y+q)[0]
                else:
                    ### dense
                    ## pseudoinverse
                    #x = (y+q) + pinvA@(b-A@(y+q))
                    # method 6:
                    x = np.linalg.lstsq(A,rhs, rcond=None)[0]
                    xcheck = scipy.sparse.linalg.lsmr(A,b)[0]
                    xhat = scipy.sparse.linalg.lsmr(A,rhs)[0]
                    xbar = xhat - (y+q)
                    alph = xcheck - xbar
                    bet = y+q - xbar
                    x=(np.dot(alph,bet)/(np.linalg.norm(alph)**2))*alph + xbar
                # update current solution
                # Dykstra: update q
                q = y+q - x
            ffeas = np.dot(c,x)
            # new point too close to old
            progressSol = np.linalg.norm(x - xprev)
            print("rp4lp:apretr(Dy): itn={:d} rnge={:.2f} eqe={:.2f} prgr={:.2f} obj={:.2f}".format(itn, bnderr, feaserr, progressSol, ffeas))
            #print("   DykstraVars:||x||={:.2f} ||y||={:.2f} ||p||={:.2f} ||q||={:.2f}".format(np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(p), np.linalg.norm(q)))
            if progressSol <= tol: #mySmall:
                status = "1:noprogress"
                iterations = itn + 1
                break
            # check error sum
            if bnderr + feaserr <= 2*tol:
                status = "1:smallerr"
                iterations = itn + 1
                break   

    elif AltProjMethod == "AARM":
        # Artacho & Campoy's averaged APM (AARM)
        alpha = 0.9
        beta = 0.9
        xA = np.copy(x)
        xB = np.copy(x)
        for itn in range(maxitn):
            for j in range(n):
                xprev[j] = x[j]
            # bound error, project x on low <= x <= upp
            for j in range(n):
                if x[j] < low[j]:
                    xA[j] = low[j]
                elif x[j] > upp[j]:
                    xA[j] = upp[j]
            xAI = 2*beta*xA - x            
            # feasibility error, project xA on Ax=b
            rhs = b + A.dot(xA) 
            if spA: # sparse
                ## method 6: line search between xcheck and xbar
                xcheck = scipy.sparse.linalg.lsmr(A,b)[0]
                xhat = scipy.sparse.linalg.lsmr(A,rhs)[0]
            else: # dense
                xcheck = np.linalg.lstsq(A,b,rcond=None)[0]
                xhat = np.linalg.lstsq(A,rhs,rcond=None)[0]
            xbar = xhat - xA
            alph = xcheck - xbar
            bet =  xA - xbar
            xB = (np.dot(alph,bet)/(np.linalg.norm(alph)**2))*alph + xbar
            xBI = 2*beta*xB - x
            # AARM operator
            x = (1-alpha)*x + alpha*xBI
            # check errors
            bnderr = sum([abs(min(x[j] - low[j], 0)) for j in range(n)])
            bnderr+= sum([abs(min(upp[j] - x[j], 0)) for j in range(n)])
            feaserr = np.linalg.norm(A.dot(x)-b) 
            progressSol = np.linalg.norm(x - xprev)
            ffeas = np.dot(c,x)
            print("rp4lp:apretr(AARM): itn={:d} rnge={:.2f} eqe={:.2f} prgr={:.2f} obj={:.2f}".format(itn, bnderr, feaserr, progressSol, ffeas))
            if progressSol <= tol: #mySmall:
                status = "1:noprogress"
                iterations = itn + 1
                break
            # check error sum
            if bnderr + feaserr <= 2*tol:
                status = "1:smallerr"
                iterations = itn + 1
                break   
            
    elif AltProjMethod == "vonNeumann": 
        # von Neumann's APM - since xstart feasible in Ax=b, 1st pr on [low,upp]
        for itn in range(maxitn):
            for j in range(n):
                xprev[j] = x[j]
            # verify bound error wrt low <= x <= upp
            bnderr = sum([abs(min(x[j] - low[j], 0)) for j in range(n)])
            bnderr += sum([abs(min(upp[j] - x[j], 0)) for j in range(n)])
            if bnderr > tol:
                # bound error, project on low <= x <= upp
                for j in range(n):
                    if x[j] < low[j]:
                        x[j] = low[j]
                    elif x[j] > upp[j]:
                        x[j] = upp[j]
            else: 
                # no bound error: exit
                status = "1:nobnderr"
                iterations = itn + 1
                break
            # verify feasibility error wrt Ax=b
            feaserr = np.linalg.norm(A.dot(x)-b)
            if feaserr > tol:
                # feasibility error, project on Ax=b
                rhs = b + A.dot(x)
                if spA:
                    # sparse least squares
                    (x,stat,itns,e1,e2,nA,cA,nx) = scipy.sparse.linalg.lsmr(A,rhs)
                else:
                    # dense least squares
                    x,res,rk,sv = np.linalg.lstsq(A, rhs, rcond=None)
            ffeas = np.dot(c,x)
            # new point too close to old
            progressSol = np.linalg.norm(x - xprev)
            print("rp4lp:apretr(vN): itn={:d} rng={:.2f} eq={:.2f} prg={:.2f} obj={:.2f}".format(itn, bnderr, feaserr, progressSol, ffeas))
            if progressSol <= tol: #mySmall:
                status = "1:noprogress"
                iterations = itn + 1
                break
            # check error sum
            if bnderr + feaserr <= 2*tol:
                status = "1:smallerr"
                iterations = itn + 1
                break

    elif AltProjMethod == "locslv":
        # a few iterations of a local solver
        LUbnd = [t for t in zip(low,upp)]
        status = "locsolve highs@linprog maxiter={}".format(maxitn)
        #res = scipy.optimize.linprog(-c, A_eq=A, b_eq=b, bounds=LUbnd, method='highs', x0=xstart, options={"disp":True})
        res = scipy.optimize.linprog(-c, A_eq=A, b_eq=b, bounds=LUbnd, method='highs-ipm', x0=xstart, options={"maxiter":maxitn, 'ipm_optimality_tolerance':tol, 'dual_feasibility_tolerance':tol, 'primal_feasibility_tolerance':tol, "disp":True})
        #status = "locsolve trust-constr@minimize maxiter={}".format(maxitn)
        #res = scipy.optimize.minimize(projObjFun, xstart, args=(xstart), bounds=LUbnd, constraints=(scipy.optimize.LinearConstraint(A,b,b)), tol=tol, options={"maxiter":maxitn, "gtol":tol, "disp":True})
        #res = scipy.optimize.minimize(projObjFun, xstart, args=(xstart), method="trust-constr", bounds=LUbnd, constraints=(scipy.optimize.LinearConstraint(A,b,b)), tol=tol, options={"maxiter":maxitn, "gtol":tol, "disp":True})
        x = res.get('x')
        bnderr = sum([abs(min(x[j] - low[j], 0)) for j in range(n)])
        bnderr += sum([abs(min(upp[j] - x[j], 0)) for j in range(n)])
        feaserr = np.linalg.norm(A.dot(x)-b)
        progressSol = np.linalg.norm(x - xprev)
        ffeas = np.dot(c,x)
            
    else:
        # APM method not found
        print("rp4lp:apretr: method {} not found".format(AltProjMethod))
        x = xstart

    tapm = time.time() - t0
        
    # bisection search in objfundir on segment [x,xproj]
    if xp is not None and AltProjMethod != "locslv":
        print("rp4lp:bisection: using segment [xapm,xproj]")
        #bnderr = sum([abs(min(x[j] - low[j],0)) for j in range(n)])
        #bnderr += sum([abs(min(upp[j] - x[j],0)) for j in range(n)])
        #feaserr = np.linalg.norm(A.dot(x) - b)
        print("rp4lp:bisection: xapm has bnderr={:f} feaserr={:f} obj={:f}".format(bnderr, feaserr, ffeas))
        x = bisectionSearch(x,xp,A,b,c,objfundir)

    tbis = time.time() - tapm - t0
            
    # compute bnderr and feaserr for new solution and distance from old
    bnderr = sum([abs(min(x[j] - low[j],0)) for j in range(n)])
    bnderr += sum([abs(min(upp[j] - x[j],0)) for j in range(n)])
    feaserr = np.linalg.norm(A.dot(x) - b)
    fopt = np.dot(c,x)
    aprcpu = time.time() - aprcpu0
    if debug:
        bnderrx = sum([abs(min(xstart[j]-low[j],0)) for j in range(n)])
        bnderrx += sum([abs(min(upp[j]-xstart[j],0)) for j in range(n)])
        feaserrx = np.linalg.norm(A.dot(xstart) - b)
        print("rp4lp:apretr: status={:s} itn={:d}".format(status,iterations))
        print("rp4lp:apretr: forg={:f} fnew={:f}".format(forg,fopt))
        print("rp4lp:apretr: bnderrold={:f} bnderrnew={:f}".format(bnderrx,bnderr))
        print("rp4lp:apretr: feaserrold={:f} feaserrnew={:f}".format(feaserrx,feaserr))
        print("rp4lp:apretr:cpu: tapm={:.2f} tbis={:.2f} ttot={:.2f}".format(tapm, tbis, aprcpu))
    return x, fopt, status, iterations, bnderr, feaserr
        

############################## main ##############################

if len(sys.argv) < 2:
    exit('syntax is [./rp4lp.py instance.dat [instance2.dat ...] ]')

## open file(s)
datFiles = sys.argv[1:]

print("rp4lp:conf: jlleps tested:", jllEPS)
print("rp4lp:conf: {0:d} runs for each (instance,jlleps)".format(runsPerEps))
print("rp4lp:conf: RPDensFactor={:f} universalConstant={:f}".format(RPDensFactor, universalConstant))
print("rp4lp:conf: org solver={:s} {:s}".format(LPSolverOrg, orgoptions))
print("rp4lp:conf: prj solver={:s} {:s}".format(LPSolverPrj, prjoptions))
print("rp4lp:conf: can do", instanceTypes)
print("rp4lp:conf:maxflow: sparseFlag={}".format(sparseFlag))
print("rp4lp:conf:diet: retrJLLMOR={} slackCoeff={}".format(retrJLLMOR, slackCoeff))

print("OUTLABELS:run,jlleps,lp,m,k,n,dens,nnz,nnzp,f*,fp,fr,avginq,maxinq,avgeq,maxeq,|x*|,|xp|,|xr|,|x*-xp|,|x*-xr|,|xp-xr|,|y|,theta,thetaest,t,tp")

# org
pname = {df:'' for df in datFiles}
dimorg = {df:0 for df in datFiles}
cols = {df:0 for df in datFiles}
constrdens = {df:0.0 for df in datFiles}
nnzorg = {df:0 for df in datFiles}
thetaest = {df:0.0 for df in datFiles}
fstar = {df:0.0 for df in datFiles}
normxstar = {df:0.0 for df in datFiles}
normdual = {df:0.0 for df in datFiles}
theta = {df:0.0 for df in datFiles}
torgd = {df:0.0 for df in datFiles}

# prj: dicts of dicts of dicts (instance, jlleps, run)
dimprj = dict()
nnzprj = dict()
fproj = dict()
fretr = dict()
xproj = dict()
xretr = dict()
normxproj = dict()
normxretr = dict()
dsp = dict()
dsr = dict()
dpr = dict()
avgineqinfeas = dict()
maxineqinfeas = dict()
avgAbinfeas = dict()
maxAbinfeas = dict()
tprj = dict()

# general averages over all instances, jlleps, runs
# org
avgAll_dimorg = 0
avgAll_cols = 0
avgAll_constrdens = 0
avgAll_nnzorg = 0
avgAll_thetaest = 0.0
avgAll_fstar = 0.0
avgAll_normxstar = 0.0
avgAll_normdual = 0.0
avgAll_theta = 0.0
avgAll_torg = 0.0
# prj
avgAll_dimprj = 0
avgAll_nnzprj = 0
avgAll_fproj = 0
avgAll_fretr = 0
avgAll_nmxpj = 0
avgAll_nmxrt = 0
avgAll_dsp = 0
avgAll_dsr = 0
avgAll_dpr = 0
avgAll_avgin = 0
avgAll_maxin = 0
avgAll_avgeq = 0
avgAll_maxeq = 0
avgAll_tprj = 0

# initialize averages over each instance
avgInst_dimprj = dict()
avgInst_nnzprj = dict()
avgInst_fproj = dict()
avgInst_fretr = dict()
avgInst_avgin = dict()
avgInst_maxin = dict()
avgInst_avgeq = dict()
avgInst_maxeq = dict()
avgInst_nmxpj = dict()
avgInst_nmxrt = dict()
avgInst_dsp = dict()
avgInst_dsr = dict()
avgInst_dpr = dict()
avgInst_tprj = dict()

# initialize averages over each set of runs for given jlleps
avgEps_dimprj = dict()
avgEps_nnzprj = dict()
avgEps_fproj = dict()
avgEps_fretr = dict()
avgEps_avgin = dict()
avgEps_maxin = dict()
avgEps_avgeq = dict()
avgEps_maxeq = dict()
avgEps_nmxpj = dict()
avgEps_nmxrt = dict()
avgEps_dsp = dict()
avgEps_dsr = dict()
avgEps_dpr = dict()
avgEps_tprj = dict()


for df in datFiles: # loop over file set

    print("rp4lp:main: handling instance", df)
    
    try:
        if df[-3:] == ".gz":
            datf = gzip.open(df, 'r')
        else:
            datf = open(df, 'r')
    except:
        print("error: can't open file {0:s}".format(df))

    # read AMPL dat file into triplets (paramname, setname, values)
    data = readDat(datf, instanceTypes)
    datf.close()

    # determine instance type
    instanceType = None
    for t in data:
        if isinstance(t[0][0], str):
            if t[0][0] == "type":
                instanceType = t[2][0]
                break

    h = 0 # solution process index for a given instance (over jlleps and runs)

    # record statistics for each instance
    dimprj[df] = dict()
    nnzprj[df] = dict()
    fproj[df] = dict()
    fretr[df] = dict()
    xproj[df] = dict()
    xretr[df] = dict()
    normxproj[df] = dict()
    normxretr[df] = dict()
    dsp[df] = dict()
    dsr[df] = dict()
    dpr[df] = dict()
    avgineqinfeas[df] = dict()
    maxineqinfeas[df] = dict()
    avgAbinfeas[df] = dict()
    maxAbinfeas[df] = dict()
    tprj[df] = dict()

    # initialize averages over each jlleps set of runs
    avgEps_dimprj[df] = dict()
    avgEps_nnzprj[df] = dict()
    avgEps_fproj[df] = dict()
    avgEps_fretr[df] = dict()
    avgEps_avgin[df] = dict()
    avgEps_maxin[df] = dict()
    avgEps_avgeq[df] = dict()
    avgEps_maxeq[df] = dict()
    avgEps_nmxpj[df] = dict()
    avgEps_nmxrt[df] = dict()
    avgEps_dsp[df] = dict()
    avgEps_dsr[df] = dict()
    avgEps_dpr[df] = dict()
    avgEps_tprj[df] = dict()
    
    # loop over jlleps values
    for jl,jlleps in enumerate(jllEPS): 

        # record statistics for each instance and jlleps
        dimprj[df][jl] = dict()
        nnzprj[df][jl] = dict()
        fproj[df][jl] = dict()
        fretr[df][jl] = dict()
        xproj[df][jl] = dict()
        xretr[df][jl] = dict()
        normxproj[df][jl] = dict()
        normxretr[df][jl] = dict()
        dsp[df][jl] = dict()
        dsr[df][jl] = dict()
        dpr[df][jl] = dict()
        avgineqinfeas[df][jl] = dict()
        maxineqinfeas[df][jl] = dict()
        avgAbinfeas[df][jl] = dict()
        maxAbinfeas[df][jl] = dict()
        tprj[df][jl] = dict()

        # loop over same jlleps
        for run in range(runsPerEps): 

            # solve org instance only once over runs and jlleps
            runOrg = False
            if h == 0: # first time instance is solved
                runOrg = True
                fstarbak = None
                xstarbak = None

            t0 = time.time()

            # solve formulations for each type
            if instanceType == "max flow":
                pname[df] = "MaxFlow"
                formulation, objfundir, objective, solution, error, cpu = maxFlow(data, t0, jlleps, runOrg, fstar=fstarbak, xstar=xstarbak)
            elif instanceType == "diet":
                pname[df] = "Diet"
                formulation, objfundir, objective, solution, error, cpu = Diet(data, t0, jlleps, runOrg, fstar=fstarbak, xstar=xstarbak)
            elif instanceType == "quantile regression":
                pname[df] = "QuantReg"
                formulation, objfundir, objective, solution, error, cpu = QuantReg(data, t0, jlleps, runOrg, fstar=fstarbak, xstar=xstarbak)
            elif instanceType == "basis pursuit":
                pname[df] = "BasisPursuit"
                formulation, objfundir, objective, solution, error, cpu = BasisPursuit(data, t0, jlleps, runOrg, fstar=fstarbak, xstar=xstarbak)
            elif instanceType == "uniform":
                pname[df] = "Uniform"
                formulation, objfundir, objective, solution, error, cpu = Uniform(data, t0, jlleps, runOrg, fstar=fstarbak, xstar=xstarbak)
            
            # retrieve solution statistics

            if runOrg:
                # record statistics from original instance 
                dimorg[df] = formulation[0]
                cols[df] = formulation[2]
                constrdens[df] = formulation[3]
                nnzorg[df] = formulation[4]
                thetaest[df] = formulation[6]
                fstar[df] = objective[0]
                xstar = solution[0]
                normxstar[df] = solution[3]
                normdual[df] = solution[9]
                theta[df] = solution[10]
                torgd[df] = cpu[0]
                # update averages about original formulation
                avgAll_dimorg += dimorg[df]
                avgAll_cols += cols[df]
                avgAll_constrdens += constrdens[df]
                avgAll_nnzorg += nnzorg[df]
                avgAll_thetaest += thetaest[df]
                avgAll_fstar += fstar[df]
                avgAll_normxstar += normxstar[df]
                avgAll_normdual += normdual[df]
                avgAll_theta += theta[df]
                avgAll_torg += torgd[df]
                # these two are for passing to subsequent calls
                fstarbak = fstar[df]
                xstarbak = xstar

            # for all runs (prj instance)
            dimprj[df][jl][h] = formulation[1]
            nnzprj[df][jl][h] = formulation[5]
            fproj[df][jl][h] = objective[1]
            fretr[df][jl][h] = objective[2]
            xproj[df][jl][h] = solution[1]
            xretr[df][jl][h] = solution[2]
            normxproj[df][jl][h] = solution[4]
            normxretr[df][jl][h] = solution[5]
            dsp[df][jl][h] = solution[6]
            dsr[df][jl][h] = solution[7]
            dpr[df][jl][h] = solution[8]
            avgineqinfeas[df][jl][h] = error[0]
            maxineqinfeas[df][jl][h] = error[1]
            avgAbinfeas[df][jl][h] = error[2]
            maxAbinfeas[df][jl][h] = error[3]
            tprj[df][jl][h] = cpu[1]

            #print("OUTLABELS:run,jlleps,lp,m,k,n,dens,nnz,nnzp,f*,fp,fr,avginq,maxinq,avgeq,maxeq,|x*|,|xp|,|xr|,|x*-xp|,|x*-xr|,|xp-xr|,|y|,theta,thetaest,t,tp")
            print("OUT:{:d},{:g},{:s},{:d},{:d},{:d},{:.4f},{:d},{:d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:.2f},{:.2f}".format(run+offset, jlleps, pname[df], dimorg[df], dimprj[df][jl][h], cols[df], constrdens[df], nnzorg[df], nnzprj[df][jl][h], fstar[df], fproj[df][jl][h], fretr[df][jl][h], avgineqinfeas[df][jl][h], maxineqinfeas[df][jl][h], avgAbinfeas[df][jl][h], maxAbinfeas[df][jl][h], normxstar[df], normxproj[df][jl][h], normxretr[df][jl][h], dsp[df][jl][h], dsr[df][jl][h], dpr[df][jl][h], normdual[df], theta[df], thetaest[df], torgd[df], tprj[df][jl][h]))
            # increase global run index over instance
            h += 1

        ## end for runPerEps

        # compute averages over a given jlleps
        avgEps_dimprj[df][jl] = int(round(sum(dimprj[df][jl].values())/runsPerEps))
        avgEps_nnzprj[df][jl] = int(round(sum(nnzprj[df][jl].values())/runsPerEps))
        avgEps_fproj[df][jl] = sum(fproj[df][jl].values()) / runsPerEps
        avgEps_fretr[df][jl] = sum(fretr[df][jl].values()) / runsPerEps
        avgEps_avgin[df][jl] = sum(avgineqinfeas[df][jl].values()) / runsPerEps
        avgEps_maxin[df][jl] = sum(maxineqinfeas[df][jl].values()) / runsPerEps
        avgEps_avgeq[df][jl] = sum(avgAbinfeas[df][jl].values()) / runsPerEps
        avgEps_maxeq[df][jl] = sum(maxAbinfeas[df][jl].values()) / runsPerEps
        avgEps_nmxpj[df][jl] = sum(normxproj[df][jl].values()) / runsPerEps
        avgEps_nmxrt[df][jl] = sum(normxretr[df][jl].values()) / runsPerEps
        avgEps_dsp[df][jl] = sum(dsp[df][jl].values()) / runsPerEps
        avgEps_dsr[df][jl] = sum(dsr[df][jl].values()) / runsPerEps
        avgEps_dpr[df][jl] = sum(dpr[df][jl].values()) / runsPerEps
        avgEps_tprj[df][jl] = sum(tprj[df][jl].values()) / runsPerEps

        print("OUTEPS:-1,{:g},{:s},{:d},{:d},{:d},{:.4f},{:d},{:d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:.2f},{:.2f}".format(jlleps, pname[df], dimorg[df], avgEps_dimprj[df][jl], cols[df], constrdens[df], nnzorg[df], avgEps_nnzprj[df][jl], fstar[df], avgEps_fproj[df][jl], avgEps_fretr[df][jl], avgEps_avgin[df][jl], avgEps_maxin[df][jl], avgEps_avgeq[df][jl], avgEps_maxeq[df][jl], normxstar[df], avgEps_nmxpj[df][jl], avgEps_nmxrt[df][jl], avgEps_dsp[df][jl], avgEps_dsr[df][jl], avgEps_dpr[df][jl], normdual[df], theta[df], thetaest[df], torgd[df], avgEps_tprj[df][jl]))

    ## end for jllEPS
    avgInst_dimprj[df] = int(round(sum(avgEps_dimprj[df].values()) / len(jllEPS)))
    avgInst_nnzprj[df] = int(round(sum(avgEps_nnzprj[df].values()) / len(jllEPS)))
    avgInst_fproj[df] = sum(avgEps_fproj[df].values()) / len(jllEPS)
    avgInst_fretr[df] = sum(avgEps_fretr[df].values()) / len(jllEPS)
    avgInst_avgin[df] = sum(avgEps_avgin[df].values()) / len(jllEPS)
    avgInst_maxin[df] = sum(avgEps_maxin[df].values()) / len(jllEPS)
    avgInst_avgeq[df] = sum(avgEps_avgeq[df].values()) / len(jllEPS)
    avgInst_maxeq[df] = sum(avgEps_maxeq[df].values()) / len(jllEPS)
    avgInst_nmxpj[df] = sum(avgEps_nmxpj[df].values()) / len(jllEPS)
    avgInst_nmxrt[df] = sum(avgEps_nmxrt[df].values()) / len(jllEPS)
    avgInst_dsp[df] = sum(avgEps_dsp[df].values()) / len(jllEPS)
    avgInst_dsr[df] = sum(avgEps_dsr[df].values()) / len(jllEPS)
    avgInst_dpr[df] = sum(avgEps_dpr[df].values()) / len(jllEPS)
    avgInst_tprj[df] = sum(avgEps_tprj[df].values()) / len(jllEPS)

    print("OUTINST:-1,-1,{:s},{:d},{:d},{:d},{:.4f},{:d},{:d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:.2f},{:.2f}".format(pname[df], dimorg[df], avgInst_dimprj[df], cols[df], constrdens[df], nnzorg[df], avgInst_nnzprj[df], fstar[df], avgInst_fproj[df], avgInst_fretr[df], avgInst_avgin[df], avgInst_maxin[df], avgInst_avgeq[df], avgInst_maxeq[df], normxstar[df], avgInst_nmxpj[df], avgInst_nmxrt[df], avgInst_dsp[df], avgInst_dsr[df], avgInst_dpr[df], normdual[df], theta[df], thetaest[df], torgd[df], avgInst_tprj[df]))
    
## end for datFiles

# compute overall averages
# org
avgAll_dimorg = int(round(avgAll_dimorg / len(datFiles)))
avgAll_cols = int(round(avgAll_cols / len(datFiles)))
avgAll_constrdens /= len(datFiles)
avgAll_nnzorg = int(round(avgAll_nnzorg / len(datFiles)))
avgAll_thetaest /= len(datFiles)
avgAll_fstar /= len(datFiles)
avgAll_normxstar /= len(datFiles)
avgAll_normdual /= len(datFiles)
avgAll_theta /= len(datFiles)
avgAll_torg /= len(datFiles)
# prj
avgAll_dimprj = int(round(sum(avgInst_dimprj.values()) / len(datFiles)))
avgAll_nnzprj = int(round(sum(avgInst_nnzprj.values()) / len(datFiles)))
avgAll_fproj = sum(avgInst_fproj.values()) / len(datFiles)
avgAll_fretr = sum(avgInst_fretr.values()) / len(datFiles)
avgAll_nmxpj = sum(avgInst_nmxpj.values()) / len(datFiles)
avgAll_nmxrt = sum(avgInst_nmxrt.values()) / len(datFiles)
avgAll_dsp = sum(avgInst_dsp.values()) / len(datFiles)
avgAll_dsr = sum(avgInst_dsr.values()) / len(datFiles)
avgAll_dpr = sum(avgInst_dpr.values()) / len(datFiles)
avgAll_avgin = sum(avgInst_avgin.values()) / len(datFiles)
avgAll_maxin = sum(avgInst_maxin.values()) / len(datFiles)
avgAll_avgeq = sum(avgInst_avgeq.values()) / len(datFiles)
avgAll_maxeq = sum(avgInst_maxeq.values()) / len(datFiles)
avgAll_tprj = sum(avgInst_tprj.values()) / len(datFiles)

print("OUTALL:-1,-1,{:s},{:d},{:d},{:d},{:.4f},{:d},{:d},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:.2f},{:.2f}".format(pname[df], avgAll_dimorg, avgAll_dimprj, avgAll_cols, avgAll_constrdens, avgAll_nnzorg, avgAll_nnzprj, avgAll_fstar, avgAll_fproj, avgAll_fretr, avgAll_avgin, avgAll_maxin, avgAll_avgeq, avgAll_maxeq, avgAll_normxstar, avgAll_nmxpj, avgAll_nmxrt, avgAll_dsp, avgAll_dsr, avgAll_dpr, avgAll_normdual, avgAll_theta, avgAll_thetaest, avgAll_torg, avgAll_tprj))


# output csv with lp,jlleps,m,k,n,densA,densTA,fp/f*,fr/f*,avginq,avgeq,tp/t of averages over runs
with open(csvName, "w") as csvf:
    print("CSV,jlleps,pname,m,k,n,dA,dTA,fp/f*,fr/f*,avgin,avgeq,tp/t*", file=csvf)
    for df in datFiles:
        for jl,jlleps in enumerate(jllEPS):
            densA = nnzorg[df] / (dimorg[df]*cols[df])
            densTA = avgEps_nnzprj[df][jl] / (avgEps_dimprj[df][jl]*cols[df])
            fpdivf = avgEps_fproj[df][jl] / fstar[df]
            frdivf = avgEps_fretr[df][jl] / fstar[df]
            tpdivt = avgEps_tprj[df][jl] / torgd[df]
            print("avg{:d},{:.2f},{:s},{:d},{:d},{:d},{:.4f},{:.4f},{:f},{:f},{:f},{:f},{:.2f}".format(runsPerEps,jlleps, pname[df], dimorg[df], avgEps_dimprj[df][jl], cols[df], densA, densTA, fpdivf, frdivf, avgEps_avgin[df][jl], avgEps_avgeq[df][jl],tpdivt), file=csvf)

print("rp4lp: all done")

########################### OBLIVION ##############################
quit()

    # # computation of ideal epsilon values
    # gammaPrime = abs(objfundir*(fstar - fproj))
    # gammaPrelim = jlleps * theta*theta * normdual
    # gammaCoeff = gammaPrime/gammaPrelim
    # gamma = gammaCoeff * jlleps * theta*theta * normdual
    # gammaEst = gammaEst = gammaCoeff * jlleps * thetaest*thetaest * normdual
    # givenGamma = 0.1
    # epsilonPrime = givenGamma / (gammaCoeff*theta*theta*normdual)
    # epsilonEst = givenGamma / (gammaCoeff*thetaest*thetaest*normdual)
    # epsilon = givenGamma / (gammaCoeff*theta*theta*normdual)
    # print("gamma={0:g} gammaEst={1:g} gamma'={2:g}".format(gamma, gammaEst, gammaPrime))
    # print("epsilon={0:g} epsilonEst={1:g} epsilonPrime={2:g}".format(epsilon, epsilonEst, epsilonPrime))


    # # make in/out adjacency lists
    # Gout = {i+1:[] for i in range(n)}
    # Gin = {i+1:[] for i in range(n)}
    # for (i,j) in AA:
    #     Gout[i].append(j)
    #     Gin[j].append(i)
        
    # # make in/out adjacency lists
    # Gout = dict()
    # Gin = dict()
    # for (i,j) in AA:
    #     if i not in Gout:
    #         Gout[i] = [j]
    #     else:
    #         Gout[i].append(j)
    #     if j not in Gin:
    #         Gin[j] = [i]
    #     else:
    #         Gin[j].append(i)

            
    # # encode matrix A for linear system Ax=0 in a dict
    # A = dict()
    # for i in range(offset,n+offset):
    #     if i not in [s,t]:
    #         # record row
    #         for j in Gout[i]:
    #             A[(i,i,j)] = 1.0
    #         for j in Gin[i]:
    #             A[(i,j,i)] = -1.0
