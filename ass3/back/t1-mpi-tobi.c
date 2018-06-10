/***********************************************************************
 Program: hellomp.c                                                  
 Author: Michael Czaja, Muttaki Aslanparcasi                                           
 matriclenumber: 4293033, 5318807                                             
 Assignment : 1                                                      
 Task: 3                                                             
 Parameters: no                                                      
 Environment variables: no                                           
                                                                     
 Description:                                                        
In this Assignment there is a method for numerical Integration. Here we use the Romberg Method combinded with MPI parallel programming.
/************************************************************************/

#include "mpi.h"    // import of the MPI definitions
#include <stdio.h>  // import of the definitions of the C IO library
#include <string.h> // import of the definitions of the string operations
#include <unistd.h> // standard unix io library definitions and declarations
#include <errno.h>  // system error numbers
#include <math.h>
#include <stdlib.h>

#define MAX_BUFFER_SIZE 1000

#define Swap(x, y)   \
    {                \
        float *temp; \
        temp = x;    \
        x = y;       \
        y = temp;    \
    }

#define MAX_DIM 12
typedef float MATRIX_T[MAX_DIM][MAX_DIM];

int commLine = 9999;
int my_rank;
int world_size;

int blocksToReadEachP;

double matrixBuf[0];

MPI_Status status;
MPI_File fh;
MPI_Info infoin;
double vectorBuf[0];
int errs = 0, err = 0;

int main(int argc, char *argv[])
{
    // -----------------------------------------------------------------[Init]--
    MATRIX_T A_local;
    double x_local[MAX_DIM];
    double b_local[MAX_DIM];

    int nOfMatrix = 8; // TODO // IN
    double epsilon;
    int maxNumberIters;
    int converged;
    char *pathToMatrix = "./testdata/Matrix_A_8x8";

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    blocksToReadEachP = nOfMatrix / world_size;
    vectorBuf[nOfMatrix];
    int numberOfElemMatrix = nOfMatrix * blocksToReadEachP;
    matrixBuf[numberOfElemMatrix];

    MPI_Comm comm;
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);

    printf("%d , outNOfMatrix(%d)\n", my_rank, nOfMatrix);
    MPI_Bcast(&nOfMatrix, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(200);
    MPI_Bcast(&epsilon, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxNumberIters, 1, MPI_INT, 0, MPI_COMM_WORLD);

    readMatrix("./testdata/Matrix_A_8x8", nOfMatrix, numberOfElemMatrix);
    /*
    Print matrix. Each process step by step
    */
    for (int i = 0; i < nOfMatrix; i++)
    {
        if (my_rank == i)
        {
            printf("(%d) -", my_rank);
            for (int x = 0; x < numberOfElemMatrix; x++)
            {
                if ((x % nOfMatrix == 0) && (i > 0))
                {
                    printf("|\n");
                }
                printf("%2f ", matrixBuf[x]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    if (my_rank == 0)
        printf("\n");

    readVectorB("./testdata/Vector_b_8x", nOfMatrix);
    for (int i = 0; i < nOfMatrix; i++)
    {
        if (my_rank == i)
        {
            printf("Vector (%d) ", my_rank);
            for (int x = 0; x < nOfMatrix; x++)
            {
                printf("%2f ", vectorBuf[x]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }

    /*
    Read_matrix("Enter the matrix", A_local, n, my_rank, p);
    Read_vector("Enter the right-hand side", b_local, n, my_rank, p);

     converged = Parallel_jacobi(A_local, x_local, b_local, n,
                                epsilon, maxNumberIters, p, my_rank);
    if (converged)
        Print_vector("The solution is", x_local, n, my_rank, p);
    else if (my_rank == 0)
        printf("Failed to converge in %d iterations\n", maxNumberIters);
    */

    MPI_Finalize();
    return 0;
}

void readMatrix(char *pathToFile, double localMatrix[], int nOfMatrix, int numberOfElemMatrix)
{

    MPI_Info_create(&infoin);
    MPI_Info_set(infoin, "access_style", "write_once,random");
    err = MPI_File_open(MPI_COMM_WORLD, pathToFile, MPI_MODE_RDWR | MPI_MODE_CREATE, infoin, &fh);
    if (err)
    {
        errs++;
        MPI_Abort(MPI_COMM_WORLD, 911);
    }
    err = MPI_File_read_ordered(fh, matrixBuf, numberOfElemMatrix, MPI_DOUBLE, &status);
    if (err)
    {
        errs++;
    }

    err = MPI_File_close(&fh);
    if (err)
    {
        errs++;
    }
}

void readVectorB(char *pathToFile, int nOfMatrix)
{

    MPI_Info_create(&infoin);
    MPI_Info_set(infoin, "access_style", "write_once,random");
    err = MPI_File_open(MPI_COMM_WORLD, pathToFile, MPI_MODE_RDWR | MPI_MODE_CREATE, infoin, &fh);
    if (err)
    {
        errs++;
        MPI_Abort(MPI_COMM_WORLD, 911);
    }
    err = MPI_File_read(fh, vectorBuf, nOfMatrix, MPI_DOUBLE, &status);
    if (err)
    {
        errs++;
    }

    err = MPI_File_close(&fh);
    if (err)
    {
        errs++;
    }
}