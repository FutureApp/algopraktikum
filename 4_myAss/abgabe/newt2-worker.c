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

/**
 * @brief Multiples two matrices.
 * 
 * @param mat1 Matrix A
 * @param mat2 Matrix B
 * @param res  Matrix C -(result)
 * @param block  Number of elems per row.
 */
void multiply1D(double *mat1, double *mat2, double *res, int block, int rank)
{
    int print = 0;
    int i, j, k;
    double temp_res = 0;
    for (i = 0; i < block; i++)
    {

        for (j = 0; j < block; j++)
        {
            for (k = 0; k < block; k++)
            {
                temp_res = 0;
                temp_res = res[j + i * block];
                res[j + i * block] = temp_res + (mat1[k + i * block] * mat2[j + k * block]);
                if (print == rank)
                    printf("%d = %f + %d * %d\n", j + i * block, temp_res, k + i * block, j + k * block);
                //printf("%f = %f + %f * %f\n", res[j + i * block], temp_res, mat1[k + i * block], mat2[j + k * block]);
            }
            if (print == rank)
                printf("Pos %d = %f\n", j + i * block, res[j + i * block]);
        }
    }
    if (print == rank)
        printf("-----------------------------\n");
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
        int sizeOfOriMatrix = -1, numberElmsOneDir;
        // sizeOfOriMatrix = 16 given 16x16 matrix > 4 x 4 sub_matrix
        MPI_Bcast(&sizeOfOriMatrix, 1, MPI_INT, 0, parent);
        numberElmsOneDir = sqrt(sizeOfOriMatrix);

        double local_send_buffer_matrix[numberElmsOneDir];
        int local_send_count[world_size];

        for (i = 0; i < world_size; i++)
            local_send_count[i] = 0;

        double local_2d_matrixA[numberElmsOneDir][numberElmsOneDir];
        double local_2d_matrixB[numberElmsOneDir][numberElmsOneDir];
        double local_2d_matrixC[numberElmsOneDir][numberElmsOneDir];

        double local_1d_matrixA[numberElmsOneDir * numberElmsOneDir];
        double local_1d_matrixB[numberElmsOneDir * numberElmsOneDir];
        double local_1d_matrixC[numberElmsOneDir * numberElmsOneDir];

        // zeroe's all matrix entry's
        for (i = 0; i < numberElmsOneDir; i++)
        {
            for (x = 0; x < numberElmsOneDir; x++)
            {
                local_2d_matrixA[x][i] = 0;
                local_2d_matrixB[x][i] = 0;
                local_2d_matrixC[x][i] = 0;

                local_1d_matrixA[i + x * numberElmsOneDir] = 0;
                local_1d_matrixB[i + x * numberElmsOneDir] = 0;
                local_1d_matrixC[i + x * numberElmsOneDir] = 0;
            }
        }

        // Get parts of the ori matrix A and B.
        MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixA, numberElmsOneDir * numberElmsOneDir, MPI_DOUBLE, 0, parent);
        MPI_Scatterv(local_send_buffer_matrix, local_send_count, local_send_count, MPI_DOUBLE, local_1d_matrixB, numberElmsOneDir * numberElmsOneDir, MPI_DOUBLE, 0, parent);

        // Create the cart-communicator
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

        int moveLeftXTimes = me / dimSize, moveUpXTimes = me % dimSize;
        int dir = 1, numberOfSendingElms = sizeOfOriMatrix;
        int disp = moveLeftXTimes;
        int rank_source, rank_dest;

        printf("        ME %d: left:%d up:%d\n", me, moveLeftXTimes, moveUpXTimes);
        printer = 0;
        if (me == printer)
        {
            printf("\nLocal C- Befors | ME-W(%d)\n", me);
            for (i = 0; i < numberElmsOneDir * numberElmsOneDir; i++)
            {
                if (i % numberElmsOneDir == 0)
                    printf("\n");
                printf("%.3f ", local_1d_matrixC[i]);
            }
            printf("\n-----------------\n");
        }

        // --------------------------------------------[ Alignment R&C ]--
        // Shifts rows left till boarder
        if (me == printer)
            printf("    Alignment Start\n");
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_1d_matrixA, numberOfSendingElms, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);

        // Shifts cols top till boarder
        dir = 0;
        disp = moveUpXTimes;
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_1d_matrixB, numberOfSendingElms, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        if (me == printer)
            printf("    Alignment End\n");

        // --------------------------------------------[ Calculate ]--
        // Makes calcs and shifts elems
        int roundsToShift = sqrt(nodesInCart);
        int k = 0, j = 0, block = numberElmsOneDir;
        disp = -1;
        if (me == printer)
            printf("    Calc start\n");
        for (x = 0; x < roundsToShift; x++)
        {
            if (me == printer)
                printf("    Round %d/%d\n", x, roundsToShift);

            for (i = 0; i < block; i++)
            {
                for (j = 0; j < block; j++)
                {
                    double partRes = 0;
                    for (k = 0; k < block; k++)
                    {
                        partRes += local_1d_matrixA[k + i * block] * local_1d_matrixB[j + k * block];
                    }
                    double temp = local_1d_matrixC[j + i * block];
                    local_1d_matrixC[j + i * block] = temp + partRes;
                }
            }

            dir = 1;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_1d_matrixA, numberOfSendingElms, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
            dir = 0;
            MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
            MPI_Sendrecv_replace(local_1d_matrixB, numberOfSendingElms, MPI_DOUBLE, rank_dest, 0,
                                 rank_source, 0, cartCom, MPI_STATUS_IGNORE);
            /*
                                 */
        }

        if (me == printer)
        {
            printf("\nLocal C- FINAL | ME-W(%d)\n", me);
            for (i = 0; i < sizeOfOriMatrix; i++)
            {
                if (i % numberElmsOneDir == 0)
                    printf("\n");
                printf("%.3f ", local_1d_matrixC[i]);
            }
            printf("\n-----------------\n");
        }

        if (me == printer)
            printf("    Calc END\n");
        MPI_Barrier(cartCom);
        exit(1);
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