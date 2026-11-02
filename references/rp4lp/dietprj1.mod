# dietprj1.mod 

param m integer, default 1;
param k integer, default 1;
param n integer, default 1;
param npm := n+m;
set K := 1..k;
set M := 1..m;
set N := 1..n;
set NM := 1..npm;
param slackCoeff >= 0, default 10.0;
param TA{K,NM} default 0;
param c{N} default 0;
param Tb{K} default 0;

var xs{NM} >= 0;

# prj diet form 1 in JEA paper: min cq + 1r : TDq + Tr = Tb, q,r>=0
minimize costprj: sum{j in N} c[j]*xs[j] + slackCoeff*sum{j in n+1..npm} xs[j];

subject to nutrientprj{i in K}: sum{j in NM} TA[i,j]*xs[j] = Tb[i];

