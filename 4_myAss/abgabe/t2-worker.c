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
    int world_size, i;

    local_matrixC[0] = 0; // This is the init value.
    MPI_File mpi_file;
    MPI_Comm parent;
    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent);
    if (parent == MPI_COMM_NULL)
        error("No parent!");
    MPI_Comm_remote_size(parent, &size);
    if (size != 1)
        error("Something's wrong with the parent");
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
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
        MPI_Sendrecv_replace(local_matrixA, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);

        // Shifts cols top till boarder
        dir = 0;
        disp = -coords[1];
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_matrixB, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        /**/
        // Makes calcs and shifts elems
        int roundsToShift = sqrt(nodesInCart);
        disp = -1;
        for (i = 0; i < roundsToShift; i++)
        {
            double c_before = local_matrixC[0];
            local_matrixC[0] = local_matrixC[0] + local_matrixA[0] * local_matrixB[0];
            dir = 1;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_matrixA, 1, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
            dir = 0;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_matrixB, 1, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        }
        // -------------------------------------------------------[ SAVE RESULT ]--
        char *pathToResultFile = "./result.double"; //PATH where to save result
        err = MPI_File_open(cartCom, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        if (err)
            printf("\nError opening the file.\n");
        MPI_File_write_ordered(mpi_file, local_matrixC, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);
        //####################################################################################
        // FINAL Condition
        int finalStatus = 0;
        flags = 0;
        MPI_Ibcast(&finalStatus, 1, MPI_INT, 0, parent, &request);
        int firstOne = 0;
        while (flags == 0)
        {
            if (firstOne == 0)
            {
                //printf("[%d] My calc is finished. Waiting for master-com now\n", me);
                firstOne++;
            }
            MPI_Test(&request, &flags, &status);
        }
    }

    //./result.double
    MPI_Finalize();
    return 0;
}