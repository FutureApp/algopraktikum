/***********************************************************************
 worker_program: t2-master.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 4
 Task: 2

 Description:
MPI worker_program which multiplies two given matrixes using the cannon-algorithm.
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

    // MPI_STUFF
    int my_rank, world_size;
    // Other stuff
    char pathToA[64] = "", pathToB[64] = "", pathToC[64] = "";
    char *ptrA, *ptrB;
    int printer = 0, flags = 0, elmMatCounter = 0, err = 0;
    int i, x, y;

    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Section to handle user-interaction and get information of ptrA and ptrB
    /*
    */
    MPI_File filo;
    char *pathsToResultFile = "./test16x16.double"; //PATH where to save result
    int times = 16 * 16;
    double localTest[times];
    double interCounterMY = 1;
    for (int i = 0; i < times; i++)
    {
        localTest[i] = interCounterMY;
        if (interCounterMY == 16)
            interCounterMY = 1;
        else
            interCounterMY++;
    }
    MPI_File_open(MPI_COMM_SELF, pathsToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &filo);
    MPI_File_write(filo, localTest, times, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&filo);
    // -------

    // DEV
    ptrA = "./A_16x16";
    ptrB = "./B_16x16";
    // ----

    // printf("Path to matrix A: %s\n", ptrA);
    // printf("Path to matrix B: %s\n", ptrB);

    // ---------------------------------------------------------------[ Load matrix A,B] -----------------
    MPI_File mpi_fileA, mpi_fileB;
    MPI_Offset fsizeA, fsizeB;
    int elmsOfMatrixA, elmsOfMatrixB;

    err = MPI_File_open(MPI_COMM_SELF, ptrA, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileA);
    MPI_File_get_size(mpi_fileA, &fsizeA);
    err = MPI_File_open(MPI_COMM_SELF, ptrB, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileB);
    MPI_File_get_size(mpi_fileB, &fsizeB);

    elmsOfMatrixA = fsizeA / (sizeof(double));
    elmsOfMatrixB = fsizeB / (sizeof(double));

    double *master_1d_matrixA = malloc(sizeof(double) * elmsOfMatrixA);
    double *master_1d_matrixB = malloc(sizeof(double) * elmsOfMatrixB);
    double *master_1d_matrixC = malloc(sizeof(double) * elmsOfMatrixB); // DO i need this
    int master_matrixDimension = (int)sqrt(elmsOfMatrixA);

    MPI_File_read(mpi_fileA, master_1d_matrixA, elmsOfMatrixA, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(mpi_fileB, master_1d_matrixB, elmsOfMatrixB, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_fileA);
    MPI_File_close(&mpi_fileB);
    // ###################################################################################################

    // -----------------------------------------------------------------[ Show matrix A & B] -------------
    if (my_rank == printer)
    {
        // printf("A\n");
        for (i = 0; i < elmsOfMatrixA; i++)
        {
            if (i % master_matrixDimension == 0)
            {

                // printf("\n");
            }
            // printf("%.3f ", master_1d_matrixA[i]);
        }
        // printf("\nB\n");
        for (i = 0; i < elmsOfMatrixA; i++)
        {
            if (i % master_matrixDimension == 0)
            {

                // printf("\n");
            }
            // printf("%.3f ", master_1d_matrixB[i]);
        }
    }

    // ###################################################################################################

    // ----------------------------------------------------------[ Spawn Worker (interComm)] -------------

    int numberOfChilds = sqrt(master_matrixDimension);
    char *worker_program = "./t2-worker-prog";
    MPI_Comm child;
    int spawnError[numberOfChilds];
    // printf("MASTER spawing childs (%d).\n", numberOfChilds);
    MPI_Comm_spawn(worker_program, MPI_ARGV_NULL, numberOfChilds, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &child, spawnError);

    // Send info to childs.
    MPI_Bcast(&master_matrixDimension, 1, MPI_INT, MPI_ROOT, child);
    // ###################################################################################################

    // ---------------------------------------------------------------[ Prepare for Scatter] -------------

    // Convert 1-d matrices to 2-d matrices
    double master_2d_matrixA[master_matrixDimension][master_matrixDimension], master_2d_matrixB[master_matrixDimension][master_matrixDimension];

    for (y = 0; y < master_matrixDimension; y++)
        for (x = 0; x < master_matrixDimension; x++)
        {
            master_2d_matrixA[x][y] = master_1d_matrixA[elmMatCounter];
            master_2d_matrixB[x][y] = master_1d_matrixB[elmMatCounter];
            elmMatCounter++;
        }

    // Create the datatype
    MPI_Datatype sub_array_type, sub_array_resized;

    int sub_matrix_size = sqrt(master_matrixDimension);
    // printf(" sub: %d", sub_matrix_size);
    int complete_array_dims[2] = {master_matrixDimension, master_matrixDimension};
    int sub_array_dims[2] = {sub_matrix_size, sub_matrix_size};
    int start_array[2] = {0, 0};
    MPI_Type_create_subarray(2, complete_array_dims, sub_array_dims, start_array, MPI_ORDER_C, MPI_DOUBLE, &sub_array_type);
    MPI_Type_commit(&sub_array_type);

    MPI_Type_create_resized(sub_array_type, 0, sub_matrix_size * sizeof(double), &sub_array_resized);
    MPI_Type_commit(&sub_array_resized);

    // Calculate displacements
    int dispList[numberOfChilds], sendList[numberOfChilds];
    int disCounter = 0;
    int disSkipper = master_matrixDimension;
    int test = numberOfChilds / 2;
    int takeBack = test;

    // printf("            TAKE BACK: %d", takeBack);

    for (i = 0; i < numberOfChilds; i++)
    {
        dispList[i] = disCounter;
        disCounter++;
        if (disCounter % test == 0)
        {
            disCounter -= takeBack;
            disCounter += disSkipper;
        }
        sendList[i] = 1;
    }

    if (my_rank == printer)
    {
        // printf("disList: ");
        for (i = 0; i < numberOfChilds; i++)
        {
            // printf("%d,", dispList[i]);
        }
        // printf("\n");
    }
    // ###################################################################################################

    // ------------------------------------------------------------------[ Scatter the data] -------------
    // Scatter matrix_A and matrix_B to child processes
    double *recv_buf[master_matrixDimension * master_matrixDimension];
    int sub_matrix_elements = master_matrixDimension * master_matrixDimension;
    // printf(" MASTER : %f ", master_2d_matrixA[1][8]);
    MPI_Scatterv(master_2d_matrixA, sendList, dispList, sub_array_resized, recv_buf, sub_matrix_elements, MPI_DOUBLE, MPI_ROOT, child);
    MPI_Scatterv(master_2d_matrixB, sendList, dispList, sub_array_resized, recv_buf, sub_matrix_elements, MPI_DOUBLE, MPI_ROOT, child);
    // ###################################################################################################
    // printf("                                                                                Master off\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(child);
    MPI_Finalize();
}