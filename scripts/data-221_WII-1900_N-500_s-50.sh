#!/bin/bash
        N=8
        trap 'exit' INT
        num=$(awk 'BEGIN{for(i=0;i<4;i+=0.5)print i}')
        num2=$(awk 'BEGIN{for(i=0;i<100;i+=10)print i}')
        for n in $num
        do
          # ===============================================
          # Beginning of inner loop.
          (
          for j in $num2
          do
                ((i=i%N)); ((i++==0)) && wait
                ../cortex/cortex -G $n -S $j -N 500 -r 0.8 -d1 10 -d2 3000 -d3 10 -before 10 -after 3010 -s 50 -WII 1900 -output _data_221-wii1900-n500-r08-s50-se-72-th-2.0 -model gp-izh &
          done
          )
          # End of inner loop.
          # ===============================================
        done
        # End of outer loop.
        exit 0
        