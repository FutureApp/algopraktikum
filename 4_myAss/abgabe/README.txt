 A4 T1
 
 Example Call:
 mpicc -o ./app1 ./t1Newass4i.c -lm && mpiexec  -n 8 ./app1 -s ffm_1280x960.gray -w 1280 -f 0 -r 5  && display -depth 8 -size 1280x960 result.gray
 
 A4 T2
 Example Call:
 mpicc -o ./t2-worker-prog ./t2-worker.c && mpicc -o ./app1 ./t2-master.c -lm && mpiexec  -n 1 ./app1 
