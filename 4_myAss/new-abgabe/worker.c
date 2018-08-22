/***********************************************************************
 Program: t2-master.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 4
 Task: 2

 Description:
MPI program which multiplies two given matrixes using the cannon-algorithm.
This is the manager-component.
/************************************************************************/
#include <sys/select.h>

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <time.h>

#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int printer = 0;
    int i, x;

    int world_size, world_rank;
    int dimOfOriMatrix, dimOfLocalMatrix;
    MPI_File mpi_file;
    MPI_Comm parent_communicator;
    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent_communicator);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // get info from parent-comm
    MPI_Bcast(&dimOfOriMatrix, 1, MPI_INT, 0, parent_communicator);
    printf(" DIM GOT: %d", dimOfOriMatrix);
    dimOfLocalMatrix = (dimOfOriMatrix / sqrt(world_size));

    // ---------------------------------------------------------- Prepare ---
    double local_send_buffer_matrix[dimOfLocalMatrix];
    int local_send_count[world_size];

    for (i = 0; i < world_size; i++)
        local_send_count[i] = 0;

    double local_2d_matrixA[dimOfLocalMatrix][dimOfLocalMatrix];
    double local_2d_matrixB[dimOfLocalMatrix][dimOfLocalMatrix];
    double local_2d_matrixC[dimOfLocalMatrix][dimOfLocalMatrix];

    double local_1d_matrixA[dimOfLocalMatrix * dimOfLocalMatrix];
    double local_1d_matrixB[dimOfLocalMatrix * dimOfLocalMatrix];
    double local_1d_matrixC[dimOfLocalMatrix * dimOfLocalMatrix];

    // zeroe's all matrix entry's
    for (i = 0; i < dimOfLocalMatrix; i++)
    {
        for (x = 0; x < dimOfLocalMatrix; x++)
        {
            local_2d_matrixA[x][i] = -1;
            local_2d_matrixB[x][i] = -1;
            local_2d_matrixC[x][i] = -1;

            local_1d_matrixA[i + x * dimOfLocalMatrix] = -1;
            local_1d_matrixB[i + x * dimOfLocalMatrix] = -1;
            local_1d_matrixC[i + x * dimOfLocalMatrix] = -1;
        }
    }

    // ###################################################################################################

    // ------------------------------------------------------------------[ Scatter the data] -------------
    // Get parts of the ori matrix A and B.
    int numberOfElmsToRev = dimOfLocalMatrix * dimOfLocalMatrix;
    printf("                Toget: %d", numberOfElmsToRev);
    MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixA, numberOfElmsToRev, MPI_DOUBLE, 0, parent_communicator);
    MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixB, numberOfElmsToRev, MPI_DOUBLE, 0, parent_communicator);
    MPI_Barrier(parent_communicator);
    printf("------------ %f ------------------\n", local_1d_matrixA[15]);
    // -----------------------------------------------------------------[ Show matrix A & B] -------------
    printer=2;
    if (world_rank == printer)
    {
        printf("    Worker A\n");
        for (i = 0; i < numberOfElmsToRev; i++)
        {
            if (i % dimOfLocalMatrix == 0)
                printf("\n");
            printf("%.3f ", local_1d_matrixA[i]);
        }
        printf("\n  Worker B\n");
        for (i = 0; i < numberOfElmsToRev; i++)
        {
            if (i % dimOfLocalMatrix == 0)
                printf("\n");
            printf("%.3f ", local_1d_matrixB[i]);
        }
    }
    else
    {
    }

    // ###################################################################################################
    MPI_Barrier(parent_communicator);
    printf("Worker off\n");
    MPI_Finalize();
}