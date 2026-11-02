#!/usr/bin/env python3

## author: Leo Liberti
## purpose: read rp4lp output (*-avg.csv), print latex tables and graphs
## date: 220208

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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

######################### global params ##########################

myZero = 1e-8
myInf = 1e30
RPDensFactor = 0.5 # set RP density at RPDensFactor * [Ax=b density]
universalConstant = 1.0
instanceTypes = ["basis pursuit", "diet", "max flow", "quantile regression"]
offset = 1
nFields = 6

sns.set_theme()

########################### functions ############################

############################## main ##############################

if len(sys.argv) < 2:
    exit('syntax is [./maketables file-avg.csv [file-avg.csv ...] ]')

t0 = time.time()

## open file(s)
csvFiles = sys.argv[1:]

for cf in csvFiles: # loop over file set

    runPerEps = dict()
    jlleps = dict()
    pname = dict()
    m = dict()
    k = dict()
    n = dict()
    dA = dict()
    dTA = dict()
    fpf = dict()
    frf = dict()
    avgin = dict()
    avgeq = dict()
    tpt = dict()
    
    print("maketables: handling file", cf)
    h = 0
    with open(cf, "r") as f:
        for line in f:
            l = line.split(',')
            ll = len(l) - 1
            if l[0] == "CSV":
                header = [hd.strip() for hd in l[1:]]
            else:
                runPerEps[h] = int(l[0][-1])
                row = l[1:]
                jlleps[h] = float(row[0])
                pname[h] = row[1]
                m[h] = int(row[2])
                k[h] = int(row[3])
                n[h] = int(row[4])
                dA[h] = float(row[5])
                dTA[h] = float(row[6])
                fpf[h] = float(row[7])
                frf[h] = float(row[8])
                avgin[h] = float(row[9])
                avgeq[h] = float(row[10])
                tpt[h] = float(row[11])
                h += 1
    reclen = h
                
    pnames = set(pname.values())
    runs = set(runPerEps.values())
    pnidx = dict()
    instances = dict()
    eps = dict()
    for pn in pnames:
        pnidx[pn] = [h for h in range(reclen) if pname[h] == pn]
        instances[pn] = sorted(list(set([m[h] for h in pnidx[pn]])))
        eps[pn] = sorted(list(set([jlleps[h] for h in pnidx[pn]])))

        
        # statistics over eps for fixed instance
        print("maketables:{0:s}: writing table+plots for fixed size".format(pn.lower()))
        ltfn = pn.lower() + "-by_jlleps.tex"
        ltf = open(ltfn, "w")
        print("%% statistics by size over jlleps", file=ltf)
        print("\\begin{tabular}{r|" + (nFields)*'r' + "}", file=ltf)
        print("$\\epsilon$ & $\\bar{f}/f^\\ast$ & \\ $\\tilde{f}/f^\\ast$ & \\texttt{avgin} & \\texttt{avgeq} & $k/m$ & $\\bar{t}/t^\\ast$ \\\\ \hline", file=ltf)
        instidx = dict()
        dfi = dict()
        for inst in instances[pn]:
            instidx[inst] = [h for h in pnidx[pn] if m[h] == inst]
            print("\\multicolumn{" + str(nFields) + "}{c}{" + pn.lower() + "-" + str(inst) + "}\\\\ \\hline", file=ltf)
            instT = []
            vjlleps = np.zeros(len(instidx[inst]))
            vfpf = np.zeros(len(instidx[inst]))
            vfrf = np.zeros(len(instidx[inst]))
            vavgin = np.zeros(len(instidx[inst]))
            vavgeq = np.zeros(len(instidx[inst]))
            vkm = np.zeros(len(instidx[inst]))
            vtpt = np.zeros(len(instidx[inst]))
            for j,h in enumerate(instidx[inst]):
                vjlleps[j] = jlleps[h]
                vfpf[j] = fpf[h]
                vfrf[j] = frf[h]
                vavgin[j] = round(avgin[h], 4)
                vavgeq[j] = round(avgeq[h], 4)
                vkm[j] = k[h] / m[h]
                vtpt[j] = tpt[h]
                instT.append((vjlleps[j],vfpf[j],vfrf[j],vavgin[j],vavgeq[j],vkm[j],vtpt[j]))
            for r in sorted(instT):
                print("{:.2f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.2f} \\\\ [-0.2em]".format(r[0],r[1],r[2],r[3],r[4],r[5],r[6]), file=ltf)
            # output plot
            dfi[inst] = dict()
            dfi[inst]['obj'] = pd.DataFrame({"jlleps":vjlleps, "fp/f*":vfpf, "fr/f*":vfrf})
            dfi[inst]['err'] = pd.DataFrame({"jlleps":vjlleps, "avgin":vavgin, "avgeq":vavgeq})
            dfi[inst]['cpu'] = pd.DataFrame({"jlleps":vjlleps, "k/m":vkm, "tp/t*":vtpt})
            for t in dfi[inst].keys():
                dfm = dfi[inst][t].melt('jlleps', var_name='cols', value_name='vals')
                plot_title = pn.lower() + "-" + str(inst) + "-" + t
                sns.pointplot(x="jlleps", y="vals", hue='cols', data=dfm).set(title=plot_title)
                plt.savefig(plot_title + ".png")
                plt.clf()
        print("\\hline \\end{tabular}", file=ltf)
        ltf.close()

        # statistics over instance for fixed eps
        print("maketables:{0:s}: writing table+plots for fixed jlleps".format(pn.lower()))
        ltfn = pn.lower() + "-by_m.tex"
        ltf = open(ltfn, "w")
        print("%% statistics by jlleps over size", file=ltf)
        print("\\begin{tabular}{r|" + (nFields)*'r' + "}", file=ltf)
        print("$m$ & $\\bar{f}/f^\\ast$ & \\ $\\tilde{f}/f^\\ast$ & \\texttt{avgin} & \\texttt{avgeq} & $k/m$ & $\\bar{t}/t^\\ast$ \\\\ \hline", file=ltf)
        epsidx = dict()
        dfe = dict()
        for e in eps[pn]:
            epsidx[e] = [h for h in pnidx[pn] if jlleps[h] == e]
            print("\\multicolumn{" + str(nFields) + "}{c}{" + pn.lower() + "-$" + str(e) + "$}\\\\ \\hline", file=ltf)
            epsT = []
            vm = np.zeros(len(epsidx[e]))
            vfpf = np.zeros(len(epsidx[e]))
            vfrf = np.zeros(len(epsidx[e]))
            vavgin = np.zeros(len(epsidx[e]))
            vavgeq = np.zeros(len(epsidx[e]))
            vkm = np.zeros(len(epsidx[e]))
            vtpt = np.zeros(len(epsidx[e]))
            for j,h in enumerate(epsidx[e]):
                vm[j] = m[h]
                vfpf[j] = fpf[h]
                vfrf[j] = frf[h]
                vavgin[j] = avgin[h]
                vavgeq[j] = avgeq[h]
                vkm[j] = k[h] / m[h]
                vtpt[j] = tpt[h]
                epsT.append((m[h],vfpf[j],vfrf[j],vavgin[j],vavgeq[j],vkm[j],vtpt[j]))
            for r in sorted(epsT):
                print("{:d} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.2f} \\\\ [-0.2em]".format(r[0],r[1],r[2],r[3],r[4],r[5],r[6]), file=ltf)
            # output plot
            dfe[e] = dict()
            dfe[e]['obj'] = pd.DataFrame({"m":vm, "fp/f*":vfpf, "fr/f*":vfrf})
            dfe[e]['err'] = pd.DataFrame({"m":vm, "avgin":vavgin, "avgeq":vavgeq})
            dfe[e]['cpu'] = pd.DataFrame({"m":vm, "k/m":vkm, "tp/t*":vtpt})
            for t in dfe[e].keys():
                dfm = dfe[e][t].melt('m', var_name='cols', value_name='vals')
                dfm['m'] = dfm['m'].astype(int)
                plot_title = pn.lower() + "-" + str(e) + "-" + t
                sns.pointplot(x="m", y="vals", hue='cols', data=dfm).set(title=plot_title)
                plt.savefig(plot_title + ".png")
                plt.clf()
        print("\\hline \\end{tabular}", file=ltf)
        ltf.close()

