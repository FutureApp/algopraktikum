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
double *local_2d_matrixA;
double *local_2d_matrixB;
double *local_2d_matrixC;
int err;

void mutex()
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(400);
}

void h_printParaQuaMatrixOfDouble(char tag, double *matrix, int sizeOfOriMatrix, int rank, int maxNodes)
{
    if (rank == 1)
        printf("\n%c", tag);
    for (int i = 0; i < (sizeOfOriMatrix * sizeOfOriMatrix); i++)
    {
        mutex();
        if (rank == i)
        {
            if (i % sizeOfOriMatrix == 0)
                printf("\n");
            printf("(%2d)%f ", i, matrix[0]);
        }
    }
}

int main(int argc, char *argv[])
{
    int world_size, i, x, y, matrixDim = -1, flags = 0, world_rank = -1;
    int printer = 0;

    MPI_File mpi_file;
    MPI_Comm parent;
    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (parent == MPI_COMM_NULL)
        error("No parent!\n");
    else
    {
        if (world_rank == printer)
            printf("Connection to father establish\n");
    }
    MPI_Comm_remote_size(parent, &size);
    if (size != 1)
        error("Something's wrong with the parent\n");
    printf("[%d]WORKER on\n", world_rank);
    /* 
    * Parallel code here.  
    * The manager is represented as the process with rank 0 in (the remote 
    * group of) MPI_COMM_parent.  If the workers need to communicate among 
    * themselves, they can use MPI_COMM_WORLD. 
    */
    MPI_Request request;
    MPI_Status status;

    if (parent != MPI_COMM_NULL)
    {
        int sizeOfOriMatrix = -1, sizeEachWtoHandle;
        MPI_Bcast(&sizeOfOriMatrix, 1, MPI_INT, 0, parent);
        sizeEachWtoHandle = sqrt(sizeOfOriMatrix);
        printf("Father calls us %d", sizeOfOriMatrix);

        double local_2d_matrixA[sizeEachWtoHandle][sizeEachWtoHandle];
        double local_2d_matrixB[sizeEachWtoHandle][sizeEachWtoHandle];
        double local_2d_matrixC[sizeEachWtoHandle][sizeEachWtoHandle];
        for (i = 0; i < sizeEachWtoHandle; i++)
            for (x = 0; x < sizeEachWtoHandle; x++)
            {
                local_2d_matrixA[x][i] = 0;
                local_2d_matrixB[x][i] = 0;
                local_2d_matrixC[x][i] = 0;
            }

        printf("[%d]WORKER off dim of matrix %d\n", world_rank, sizeEachWtoHandle);
        printf("EXIT\n");

        // Receive matrix data
        double *result_matrix = (double *)calloc(sizeOfOriMatrix, sizeof(double));
        double dummy_matrix[sizeOfOriMatrix];
        //TODO SCATTERV
        MPI_Barrier(parent);
        exit(1);
        while (flags == 0)
            MPI_Test(&request, &flags, &status);
        // CALC
        //####################################################################################
        MPI_Comm cartCom;
        int nodesInCart;
        int me;
        int coords[2];
        int per[2] = {1, 1};
        int dimSize = sqrt(world_size); // !!!!!! Because of the worldsize, we know what number the dim is.
        int dims[2] = {dimSize, dimSize};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims,
                        per, 1, &cartCom);

        MPI_Comm_rank(cartCom, &me);
        MPI_Cart_coords(cartCom, me, 2, coords);
        MPI_Comm_size(cartCom, &nodesInCart);

        //Control var.
        int dir = 1;
        int disp = -coords[0];
        int rank_source, rank_dest;

        // --------------------------------------------[ Alignment R&C ]--
        // Shifts rows left till boarder
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_2d_matrixA, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);

        // Shifts cols top till boarder
        dir = 0;
        disp = -coords[1];
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_2d_matrixB, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        /**/
        // Makes calcs and shifts elems
        int roundsToShift = sqrt(nodesInCart);
        disp = -1;
        for (i = 0; i < roundsToShift; i++)
        {
            double c_before = local_2d_matrixC[0][0];
            local_2d_matrixC[0][0] = local_2d_matrixC[0][0] + local_2d_matrixA[0][0] * local_2d_matrixB[0][0];
            dir = 1;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_2d_matrixA, 1, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
            dir = 0;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_2d_matrixB, 1, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        }
        // -------------------------------------------------------[ SAVE RESULT ]--
        double *c_matrix = malloc(sizeof(double) * world_size);
        MPI_Gather(
            local_2d_matrixC,
            1,
            MPI_DOUBLE,
            c_matrix,
            1,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);
        if (me == 0)
        {
            char *pathToResultFile = "./result.double"; //PATH where to save result
            err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
            if (err)
                printf("\nError opening the file.\n");
            MPI_File_write(mpi_file, c_matrix, world_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
            MPI_File_close(&mpi_file);
        }
        MPI_Barrier(cartCom);
        //####################################################################################
        MPI_Barrier(parent);
    }
    MPI_Finalize();
    return 0;
}