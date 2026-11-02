#!/bin/sh

if [ "$1" = "" ]; then
   echo "run with any command line to generate a whole lot of instances"
   exit 1
fi

#M="basispursuit diet maxflow quantreg"
#M="maxflow"
M="uniform"

basispursuitm="0500 1000 1500 2000 2500"
basispursuitn="3000"

dietm="0500 1000 1500 2000 2500"
dietn="400"

maxflowm="0502 1002 1502 2002 2502"
maxflown="100" # dummy

quantregm="0500 1000 1500 2000 2500"
quantregn="200"

uniformm="500 1000 1500 2000 2500"
uniformn="500" # add to m

for s in $M ; do
    echo "generating $s instances"
    if [ "$s" == "basispursuit" ]; then
	# m and n are swapped in basispursuit cmd line
	mlist="$basispursuitn"
	nlist="$basispursuitm"
    elif [ "$s" == "diet" ]; then
	mlist="$dietm"
	nlist="$dietn"
    elif [ "$s" == "maxflow" ]; then
	mlist="$maxflowm"
	nlist="$maxflown"
    elif [ "$s" == "quantreg" ]; then
	mlist="$quantregm"
	nlist="$quantregn"
    elif [ "$s" == "uniform" ]; then
	mlist="$uniformm"
	nlist="$uniformn"
    fi
    for m in $mlist ; do
	for n in $nlist ; do
	    if [ "$s" == "uniform" ]; then
  		nn=$(( $m + $n ))
		echo "./${s}gen.py $m $nn"
		./${s}gen.py $m $nn
	    else
		echo "./${s}gen.py $m $n"
		./${s}gen.py $m $n		
	done
    done
done

gzip *.dat
mv *.dat.gz dat/
