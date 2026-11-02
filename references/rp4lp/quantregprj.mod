# quantreg.mod

param m integer, default 1;
param k integer, default 1;
param p integer, default 2;
set K := 1..k;
set M := 1..m;
set P := 1..p-1;
set C := 1..p-1+2*m;
param TA{K,C} default 0;
param Tb{K} default 0;
param tau >= 0, <= 1, default 0.1;

var beta{P};
var up{M} >= 0;
var um{M} >= 0;

minimize error: tau*sum{i in M} up[i] + (1-tau)*sum{i in M} um[i];

subject to quantileprj{i in K}:
  sum{j in P} TA[i,j]*beta[j] + sum{j in M} TA[i,j+p-1]*up[j] -
    sum{j in M} TA[i,j+p-1+m]*um[j] = Tb[i];


