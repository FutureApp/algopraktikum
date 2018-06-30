/***********************************************************************
 Program: my-mpi-jacobi.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 3
 Task: 2

 Description:
MPI program that solves a set of linear equations Ax = b with the Jacobi method that
converges if the distance between the vectors x^(k) and x^(k+1) is small enough. (Calc by Cols)
/************************************************************************/
#include <stdio.h>
#include "mpi.h"
#include <math.h>

#include <float.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#define SIZE_E 5
#define SIZE_D SIZE_E *SIZE_E

void h_rootPrintHelp(int my_rank);
void h_setAndCheckParams(int argc, char *argv[]);

char *pathToSrcPic;      // IN - -s
double numberOfFilterTo; // IN - -f

int my_rank, world_size; //MPI-STUFF
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Status status;
    MPI_Request ch1; // o <------ x

    int picHeightBIG = 5;
    int sendSize = picHeightBIG * 2;
    unsigned char *packLeftBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packLeftBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    int i = 0;
    for (i = 0; i < picHeightBIG * 2; i++)
    {
        packLeftBlockToSend[i] = my_rank + 10;
        packRightBlockToSend[i] = my_rank + 10;
        packLeftBlockToRecv[i] = my_rank + 10;
        packRightBlockToRecv[i] = my_rank + 10;
    }
    if (my_rank == 0)
        MPI_Recv_init(packRightBlockToRecv, sendSize, MPI_UNSIGNED_CHAR, my_rank + 1, 1, MPI_COMM_WORLD, &ch1);
    if (my_rank == 1)
        MPI_Send_init(packLeftBlockToSend, sendSize, MPI_UNSIGNED_CHAR, my_rank - 1, 1, MPI_COMM_WORLD, &ch1);

    unsigned char revBuf = packRightBlockToRecv[0];
    unsigned char sendBuf = packLeftBlockToSend[0];

    printf("Before: [%d] send(%u) rev(%u)\n", my_rank, packLeftBlockToSend[0], packRightBlockToRecv[0]);
    MPI_Start(&ch1);
    MPI_Wait(&ch1, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    if (my_rank == 0)
        printf("---\n", my_rank, packLeftBlockToSend[0], packRightBlockToRecv[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printf("After:[%d] send(%u) rev(%u)\n", my_rank, packLeftBlockToSend[0], packRightBlockToRecv[0]);

    MPI_Finalize(); // finalizing MPI interface
}
