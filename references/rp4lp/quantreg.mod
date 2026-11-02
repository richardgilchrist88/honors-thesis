# quantreg.mod

param m integer, default 1;
param p integer, default 2;
set M := 1..m;
set P := 1..p;
param D{M,P} default 0;
param bidx integer, default p;
set P0 := P diff {bidx};
param tau >= 0, <= 1, default 0.1;

var beta{P0};
var up{M} >= 0;
var um{M} >= 0;

minimize error: tau*sum{i in M} up[i] + (1-tau)*sum{i in M} um[i];

subject to quantile{i in M}:
  sum{j in P0} D[i,j]*beta[j] + up[i] - um[i] = D[i,bidx];


