
----------------------------------------------------------------------------------- [ Task 1] --
HELPME FOR ASSIGNMENT3
/***********************************************************************
 Program: my-mpi-jacobi.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 3
 Task: 1
 Description:
MPI program that solves a set of linear equations Ax = b with the Jacobi method that
converges if the distance between the vectors x^(k) and x^(k+1) is small enough.
/************************************************************************/

HOWTO Start the app

mpicc -o ./app1 ./my-mpi.c && mpiexec -f ./hosts -n 4 ./app1 -m Matrix_A_8x8 -v Vector_b_8x -e 0.0000000001





----------------------------------------------------------------------------------- [ Task 2] -- 
Ist im moment noch nicht lauffähig.
file name < 2my-mpi.c >