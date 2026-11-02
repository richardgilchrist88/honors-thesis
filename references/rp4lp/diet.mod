# diet.mod

param m integer, default 1;
param n integer, default 1;
set M := 1..m;
set N := 1..n;
param D{M,N} default 0;
param c{N} default 0;
param b{M} default 0;

var x{N} >= 0;

minimize cost: sum{j in N} c[j]*x[j];

subject to nutrient{i in M}: sum{j in N} D[i,j]*x[j] >= b[i];

