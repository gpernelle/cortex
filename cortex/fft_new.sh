
#!/bin/bash
# nested-loop.sh: Nested "for" loops.
N=8
trap "exit" INT
num=$(awk 'BEGIN{for(i=0;i<15;i+=0.1)print i}')
num2=$(awk 'BEGIN{for(i=0;i<400;i+=1)print i}')
for n in $num
do
  # ===============================================
  # Beginning of inner loop.
  (
  for j in $num2
  do
	((i=i%N)); ((i++==0)) && wait
	./cortex -G $n -S $j -N 500 -d1 10 -d2 6000 -d3 10 -before 10 -after 6010 -s 60 -WII 1400 -output _cortex_fft_1 &
  done
  )
  # End of inner loop.
  # ===============================================
done
# End of outer loop.
exit 0
