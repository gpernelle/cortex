#!/bin/bash
# nested-loop.sh: Nested "for" loops.
thr=+0
trap "exit" INT
parallel --bar -j $thr --header : python diagrams_1.py -sWII {WII} -N {N}  ::: N 100 400 ::: WII 10 50 0  
exit 0
