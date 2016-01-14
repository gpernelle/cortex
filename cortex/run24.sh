#!/bin/bash
# nested-loop.sh: Nested "for" loops.
N=8
trap "exit" INT
num=$(awk 'BEGIN{for(i=0;i<7;i+=0.5)print i}')
num2=$(awk 'BEGIN{for(i=0;i<200;i+=10)print i}')
for n in $num
do
  # ===============================================
  # Beginning of inner loop.
  (
  for j in $num2
  do
        ((i=i%N)); ((i++==0)) && wait
        ./cortex -G $n -S $j -N 500 -r 0.8 -d1 10 -d2 3000 -d3 10 -before 10 -after 3010 -s 60 -WII 1000 -output _data_24-wii900-n500-r08-s60 -model gp-izh &
  done
  )
  # End of inner loop.
  # ===============================================
done
# End of outer loop.
exit 0
