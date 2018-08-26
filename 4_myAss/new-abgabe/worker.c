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

/**
 * @brief Calcs the result of multiplying two nxn-matrixes.
 * 
 * @param matrix_a Matrix A
 * @param matrix_b Matrix B
 * @param matrix_c The result is calculated in-place.
 * @param dimOfQuadMatrix s
 */
void seq_MatrixMulti(double *matrix_a, double *matrix_b, double *matrix_c, int dimOfQuadMatrix)
{
    int i, j, k;
    for (i = 0; i < dimOfQuadMatrix; i++)
        for (j = 0; j < dimOfQuadMatrix; j++)
            for (k = 0; k < dimOfQuadMatrix; k++)
                matrix_c[i * dimOfQuadMatrix + j] += matrix_a[i * dimOfQuadMatrix + k] * matrix_b[k * dimOfQuadMatrix + j];
}

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
    //printf(" DIM GOT: %d", dimOfOriMatrix);
    dimOfLocalMatrix = sqrt(world_size);

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
            local_2d_matrixA[x][i] = 0;
            local_2d_matrixB[x][i] = 0;
            local_2d_matrixC[x][i] = 0;

            local_1d_matrixA[i + x * dimOfLocalMatrix] = 0;
            local_1d_matrixB[i + x * dimOfLocalMatrix] = 0;
            local_1d_matrixC[i + x * dimOfLocalMatrix] = 0;
        }
    }

    // ###################################################################################################

    // ------------------------------------------------------------------[ Scatter the data] -------------
    // Get parts of the ori matrix A and B.
    int numberOfElmsToRev = dimOfLocalMatrix * dimOfLocalMatrix;
    MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixA, numberOfElmsToRev, MPI_DOUBLE, 0, parent_communicator);
    MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixB, numberOfElmsToRev, MPI_DOUBLE, 0, parent_communicator);
    //MPI_Barrier(parent_communicator);
    //printf("------------ %f ------------------\n", local_1d_matrixA[15]);
    // -----------------------------------------------------------------[ Show matrix A & B] -------------
    /*
    for (i = 0; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(300);
        if (world_rank == i)
        {
            printf("\n------------------------- %d A ----------------", world_rank);
            for (x = 0; x < numberOfElmsToRev; x++)
            {
                if (x % dimOfLocalMatrix == 0)
                    printf("\n");
                printf("%.3f ", local_1d_matrixA[x]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(300);
    }*/
    // ###################################################################################################

    // ----------------------------------------------------------------- [ cartesian - typo] -------------
    if (world_rank == printer)
    {
        printf("\n Creating the cart-topo.");
    }
    MPI_Comm cartCom;
    int nodesInCart;
    int card_rank;
    int coords[2];
    int per[2] = {1, 1};
    int dimSize = sqrt(world_size); // !!!!!! Because of the worldsize, we know what number the dim is.
    int dims[2] = {dimSize, dimSize};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, per, 1, &cartCom);
    MPI_Comm_rank(cartCom, &card_rank);
    MPI_Cart_coords(cartCom, card_rank, 2, coords);
    MPI_Comm_size(cartCom, &nodesInCart);

    int bufSend = world_rank;
    int dir, disp, rank_source, rank_dest;
    dir = 1;
    disp = 1;
    int shiftLeftXTimes = coords[0], shiftTopXTimes = coords[1];

    printf("\nWORLD_RANK: %d , LEFT: %d ; TOP: %d", world_rank, shiftLeftXTimes, shiftTopXTimes);
    MPI_Cart_shift(cartCom, dir, shiftLeftXTimes, &rank_source, &rank_dest);
    MPI_Sendrecv_replace(local_1d_matrixA, numberOfElmsToRev, MPI_DOUBLE, rank_source, 99, rank_dest,
                         MPI_ANY_TAG, cartCom, MPI_STATUS_IGNORE);

    dir = 0;
    MPI_Cart_shift(cartCom, dir, shiftTopXTimes, &rank_source, &rank_dest);
    MPI_Sendrecv_replace(local_1d_matrixB, numberOfElmsToRev, MPI_DOUBLE, rank_source, 99, rank_dest,
                         MPI_ANY_TAG, cartCom, MPI_STATUS_IGNORE);

    MPI_Barrier(cartCom);
    usleep(300);
    if (printer == card_rank)
        printf("\n\n\n");
    for (i = 0; i < nodesInCart; i++)
    {
        if (i == card_rank)
        {
            printf("\nWORLD_RANK: %d , BUF: %d", world_rank, bufSend);
        }
        MPI_Barrier(cartCom);
        usleep(300);
    }
    if (printer == card_rank)
        printf("\n\n\n");
    MPI_Barrier(cartCom);
    usleep(300);
    exit(1);

    for (i = 0; i < world_size; i++)
    {
        if (world_rank == i)
        {
            printf("\n ALIVE - MY WORLD RANK: %d", world_rank);
        }
    }

    // ###################################################################################################

    // ------------------------------------------------------------------------[ Alignment ] -------------

    // Prints matrix
    for (i = 4; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        //usleep(300);
        if (card_rank == i)
        {
            printf("\n------------------------- %d A After-Alig----------------", card_rank);
            for (x = 0; x < numberOfElmsToRev; x++)
            {
                if (x % dimOfLocalMatrix == 0)
                    printf("\n");
                printf("%.3f ", local_1d_matrixA[x]);
            }
        }
    } /**/
    // ###################################################################################################

    // -------------------------------------------------------------------------[ Calculate] -------------
    /*
    if (card_rank== printer)
        printf("\n    Calculation Starts");
    // first time:
    seq_MatrixMulti(local_1d_matrixA, local_1d_matrixB, local_1d_matrixC, dimOfLocalMatrix);
    // loop vor dimSize_Cart-1 times, now.
    disp = +1;
    for (i = 0; i < dimSize_Cart - 1; i++)
    {
        dir = 1;
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        printf("\n        ** ALIVE %d ** %d (%d > %d)\n", me, numberOfElmsToRev, rank_source, rank_dest);
        MPI_Sendrecv_replace(local_1d_matrixA, numberOfElmsToRev, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        dir = 0;
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_1d_matrixB, numberOfElmsToRev, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        MPI_Barrier(cartCom);
        seq_MatrixMulti(local_1d_matrixA, local_1d_matrixB, local_1d_matrixC, dimOfLocalMatrix);
        
    }
        */
    // ###################################################################################################

    // -------------------------------------------------------------------- [ Show result C] -------------

    printer = 0;
    printf("\n                                                        ####[%d] Worker off\n", world_rank);
    /**/
    for (i = 0; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(300);
        if (world_rank == i)
        {
            printf("\n------------------------- %d C ----------------", world_rank);
            for (x = 0; x < numberOfElmsToRev; x++)
            {
                if (x % dimOfLocalMatrix == 0)
                    printf("\n");
                printf("%.3f ", local_1d_matrixC[x]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(300);
    }
    // ###################################################################################################

    // -----------------------------------------------------------------[ Show matrix A & B] -------------
    // ###################################################################################################
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(parent_communicator);
    MPI_Finalize();
}