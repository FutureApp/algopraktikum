/***********************************************************************
 Program: t2-worker.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 4
 Task: 2

 Description:
MPI program which multiplies two given matrixes using the cannon-algorithm. 
This is the worker-component.
/************************************************************************/
#include <sys/select.h>

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

void error(char *mes) { printf("%s", mes); }
int size, me, matrixDim;
double *local_matrixA;
double *local_matrixB;
double *local_matrixC;
int err;

void mutex()
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(400);
}

void h_printParaQuaMatrixOfDouble(char tag, double *matrix, int sizeOfMatrix, int rank, int maxNodes)
{
    if (rank == 1)
        printf("\n%c", tag);
    for (int i = 0; i < (sizeOfMatrix * sizeOfMatrix); i++)
    {
        mutex();
        if (rank == i)
        {
            if (i % sizeOfMatrix == 0)
                printf("\n");
            printf("(%2d)%f ", i, matrix[0]);
        }
    }
}

int main(int argc, char *argv[])
{

    local_matrixA = malloc(sizeof(double) * 1);
    local_matrixB = malloc(sizeof(double) * 1);
    local_matrixC = malloc(sizeof(double) * 1);
    local_matrixC[0] = 0; // This is the init value.

    MPI_Comm parent;
    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent);
    if (parent == MPI_COMM_NULL)
        error("No parent!");
    MPI_Comm_remote_size(parent, &size);
    if (size != 1)
        error("Something's wrong with the parent");

    /* 
    * Parallel code here.  
    * The manager is represented as the process with rank 0 in (the remote 
    * group of) MPI_COMM_PARENT.  If the workers need to communicate among 
    * themselves, they can use MPI_COMM_WORLD. 
    */
    int flags = 0;
    MPI_Request request;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    // Check if this process is a spawned one and if so get parent CPU rank
    if (parent != MPI_COMM_NULL)
    {
        int *master_ori_matrixA, *master_ori_matrixB;
        MPI_Ibcast(&matrixDim, 1, MPI_INT, 0, parent, &request);
        MPI_Iscatter(master_ori_matrixA, 1, MPI_DOUBLE, local_matrixA, 1, MPI_DOUBLE, 0, parent, &request);
        MPI_Iscatter(master_ori_matrixB, 1, MPI_DOUBLE, local_matrixB, 1, MPI_DOUBLE, 0, parent, &request);
        local_matrixC[0] = 0; // This is the init value.
        flags = 0;
        while (flags == 0)
        {
            MPI_Test(&request, &flags, &status);
        }
        // FINAL Condition
        int finalStatus = 0;
        flags = 0;
        MPI_Ibcast(&finalStatus, 1, MPI_INT, 0, parent, &request);
        int firstOne = 0;
        while (flags == 0)
        {
            if (firstOne == 0)
            {
                printf("[%d] My calc is finished. Waiting for master-com now\n", me);
                firstOne++;
            }
            MPI_Test(&request, &flags, &status);
        }
    }
    MPI_Finalize();
    return 0;
}