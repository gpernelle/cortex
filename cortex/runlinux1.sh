
#!/bin/bash
# nested-loop.sh: Nested "for" loops.
N=8
trap "exit" INT
num=$(awk 'BEGIN{for(i=0;i<30;i+=0.2)print i}')
num2=$(awk 'BEGIN{for(i=0;i<400;i+=1)print i}')
i=0
for n in $num
do
  # ===============================================
  # Beginning of inner loop.
  (
  for j in $num2
  do
	if (( i % N == 0 )); then
        	wait
    	fi
    	i=$(( $i + 1 ))
	./cortexlinux -G $n -S $j -N 500 -d1 10 -d2 6000 -d3 10 -before 10 -after 6010 -s 60 -WII 500 -output _cortex_fft_2_wii_500 &
  done
  )
  # End of inner loop.
  # ===============================================
done
# End of outer loop.
exit 0
