 A4 T1
 
 Example Call:
 mpicc -o ./app1 ./t1Newass4i.c -lm && mpiexec  -n 8 ./app1 -s ffm_1280x960.gray -w 1280 -f 0 -r 5  
 
 A4 T2
 Example Call:
 mpicc -o ./test-worker-prog ./test-worker.c && mpicc -o ./app1 ./test-master.c -lm && mpiexec  -n 1 ./app1 -a a4x4 -b a4x4 -c r4x4.double
