
#!/bin/bash
# nested-loop.sh: Nested "for" loops.
N=8
trap "exit" INT
num=$(awk 'BEGIN{for(i=0;i<7;i+=0.2)print i}')
num2=$(awk 'BEGIN{for(i=0;i<200;i+=2)print i}')
for n in $num
do
  # ===============================================
  # Beginning of inner loop.
  (
  for j in $num2
  do
	((i=i%N)); ((i++==0)) && wait
	./cortex -G $n -S $j -N 2500 -r 0.8 -d1 10 -d2 6000 -d3 10 -before 10 -after 6010 -s 60 -WII 300 -output _data_8-cc-wii300-n2500-r08 -model cc-izh &
  done
  )
  # End of inner loop.
  # ===============================================
done
# End of outer loop.
exit 0
