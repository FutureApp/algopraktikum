Rework of E4T2
Tested with 16,64 and 256 matrices [is working]. Calculation of 64, 256 need some time.
 
 Execute the following command and then follow the instructions: 
 mpicc -o ./t2-worker-prog ./worker.c -lm && mpicc -o ./app1 ./master.c -lm && mpiexec  -n 1 ./app1 
 
 