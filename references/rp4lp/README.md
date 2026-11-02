# rp4lp
research code used to write an academic paper

The RP4LP code (rp4lp.py) was written for two papers:
- numerical evaluation in ../../../papers/jll/computation/lp
- testing a strong solution retrieval with alternating projection

HOWTO:
- for all executable files (*.py, *.sh): run without arguments to obtain minimal help
- generate random instances from maxflow, diet, quantreg, basispursuit with corresponding *gen.py file
- edit rp4lp.py: set jllEPS (epsilon values to test) and runsPerEps (how many times a projected RP is solved with different projectors, then result is average over all runs) to define experiments
- run ./rp4lp.py on file set "dat/[structure]-[sizes].dat.gz" as args, redirecting output to [structure]-[timestamp].log
- mv rp4lp.csv to [structure]-[timestamp]-avg.csv
- ./run_mktables [structure]-[timestamp]-avg.csv
  with additional command will copy latex tables and png graphs to
    /Users/liberti/maths/dr2/papers/jll/computation/lp
- whole pipeline is automated with batch-rp4lp.sh
