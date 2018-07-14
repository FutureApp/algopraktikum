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

int my_rank, world_size; //MPI-STUFF
char *pathmaster_ori_matrixA;
char *pathmaster_ori_matrixB;
char *pathmaster_ori_matrixC;

int err, i;

void mutex()
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(400);
}

void h_rootPrintHelp(int my_rank);
void h_setAndCheckParams(int argc, char *argv[]);

void seq_MatrixMulti(double *matrix_a, double *matrix_b, double *matrix_c, int size);
void MatrixMatrixMultiply(int n, double *a, double *b, double *c,
                          MPI_Comm comm);

void h_printQuaMatrixOfDouble(char tag, double *matrix, int sizeOfMatrix, int rank, int rankToPrint)
{
    if (rank == rankToPrint)
    {
        printf("\n[%d]%c", rank, tag);
        for (int i = 0; i < (sizeOfMatrix * sizeOfMatrix); i++)
        {
            if (i % sizeOfMatrix == 0)
                printf("\n");
            printf("(%3d)%3.3f ", i, matrix[i]);
        }
        if (rank == 0)
            printf("\n");
    }
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
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    h_setAndCheckParams(argc, argv);
    MPI_File mpi_fileA;
    MPI_File mpi_fileB;
    MPI_Offset fsizeA;
    MPI_Offset fsizeB;
    int elemsToHandleA, elemsToHandleB;

    err = MPI_File_open(MPI_COMM_SELF, pathmaster_ori_matrixA, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileA);
    MPI_File_get_size(mpi_fileA, &fsizeA);
    err = MPI_File_open(MPI_COMM_SELF, pathmaster_ori_matrixB, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileB);
    MPI_File_get_size(mpi_fileB, &fsizeB);

    elemsToHandleA = fsizeA / (sizeof(double));
    elemsToHandleB = fsizeB / (sizeof(double));
    if (my_rank == 0)
        printf("Elms per row <%d>\n", elemsToHandleA);

    double *master_ori_matrixA = malloc(sizeof(double) * elemsToHandleA);
    double *master_ori_matrixB = malloc(sizeof(double) * elemsToHandleB);
    double *master_ori_matrixC = malloc(sizeof(double) * elemsToHandleB);

    int matrixDim = (int)sqrt(elemsToHandleA);

    MPI_File_read(mpi_fileA, master_ori_matrixA, elemsToHandleA, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(mpi_fileB, master_ori_matrixB, elemsToHandleB, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_fileA);
    MPI_File_close(&mpi_fileB);

    mutex();
    h_printQuaMatrixOfDouble('A', master_ori_matrixA, matrixDim, my_rank, 0);
    mutex();

    if (my_rank == 0)
        printf("Elms per row <%d>\n", matrixDim);

    double *local_matrixA = malloc(sizeof(double) * 1);
    double *local_matrixB = malloc(sizeof(double) * 1);
    double *local_matrixC = malloc(sizeof(double) * 1);

    MPI_Scatter(master_ori_matrixA, 1, MPI_DOUBLE, local_matrixA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(master_ori_matrixB, 1, MPI_DOUBLE, local_matrixB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_matrixC[0] = 0; // This is the init value.

    MPI_Comm cartCom;
    int nodesInCart;
    int me;
    int coords[2];
    int per[2] = {1, 1};
    int dims[2] = {4, 4};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims,
                    per, 1, &cartCom);

    MPI_Comm_rank(cartCom, &me);
    MPI_Cart_coords(cartCom, me, 2, coords);
    MPI_Comm_size(cartCom, &nodesInCart);

    /*for (i = 0; i < world_size; i++)
    {
        if (my_rank == i)
            printf("[%d] (%d,%d)\n", my_rank, coords[0], coords[1]);
        mutex();
    }*/

    /*for (i = 0; i < world_size; i++)
    {
        if (my_rank == i)
            printf("[%d] des %d\n", my_rank, (coords[0]));
        mutex();
    }*/

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
    if (my_rank == 1)
        printf("Ready to play\n");
    mutex();
    for (i = 0; i < roundsToShift; i++)
    {
        double c_before = local_matrixC[0];
        local_matrixC[0] = local_matrixC[0] + local_matrixA[0] * local_matrixB[0];
        dir = 1;
        if (me == 0)
            printf("[%d-%d] calc: %f = %f + %f * %f\n", me, i, local_matrixC[0], c_before, local_matrixA[0], local_matrixB[0]);
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_matrixA, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);
        dir = 0;
        MPI_Cart_shift(cartCom, dir, disp, &rank_source, &rank_dest);
        MPI_Sendrecv_replace(local_matrixB, 1, MPI_DOUBLE, rank_dest, 0,
                             rank_source, 0, cartCom, MPI_STATUS_IGNORE);

        mutex();
    }

    if (my_rank == 0)
        printf("A B C\n");
    mutex();
    for (i = 0; i < world_size; i++)
    {
        mutex();
        if (my_rank == i)
        {
            printf("[%d] %f %f %f\n", my_rank, local_matrixA[0], local_matrixB[0], local_matrixC[0]);
        }
    }

    // -------------------------------------------------------[ CODE ]--

    // ---------------------------------------------[ SHOW FIN MATRIX ]--

    mutex();
    h_printParaQuaMatrixOfDouble('A', local_matrixA, matrixDim, me, matrixDim * matrixDim);
    h_printParaQuaMatrixOfDouble('B', local_matrixB, matrixDim, me, matrixDim * matrixDim);
    h_printParaQuaMatrixOfDouble('C', local_matrixC, matrixDim, me, matrixDim * matrixDim);
    mutex();

    // -------------------------------------------------------[ SAVE RESULT ]--
    MPI_File mpi_file;
    char *pathToResultFile = pathmaster_ori_matrixC; //PATH where to save result

    err = MPI_File_open(cartCom, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
    if (err)
        printf("\nError opening the file.\n");
    MPI_File_write_ordered(mpi_file, local_matrixC, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);
    mutex();
    if (me == 1)
        printf("\nResult saved. Check < %s >.\n", pathToResultFile);
    mutex();
    MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
    MPI_File_read(mpi_file, master_ori_matrixC, elemsToHandleB, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    printf("%d ", elemsToHandleB);
    mutex();
    h_printQuaMatrixOfDouble('R', master_ori_matrixC, sqrt(elemsToHandleB), me, 1);
    MPI_Finalize(); // finalizing MPI interface
}

/**
 * @brief Prints the help message. Only root (rank=0) will print the message.
 * 
 * @param my_rank  Rank of node. 
 */
void h_rootPrintHelp(int my_rank)
{
    if (my_rank == 0)
    {
        printf("------------------------------------[ HELP ]\n");
        printf("*Parameter -a <path to picture>   :    Path to matrix a.\n");
        printf("*Parameter -b <path to picture>   :    Path to matrix b.  \n");
        printf("\n");
        printf("Example call:\n");
        printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
    }
}
/**
 * @brief Check end sets the needed values.
 * 
 * @param argc  Count of values.
 * @param argv  The values.
 */
void h_setAndCheckParams(int argc, char *argv[])

{
    int index;
    int c;
    int man_a = -1;
    int man_b = -1;
    int man_c = -1;

    opterr = 0;
    while ((c = getopt(argc, argv, "ha:b:c:")) != -1)
        switch (c)
        {
        case 'h':
            h_rootPrintHelp(my_rank);
            exit(0);
            break;
        case 'a':
            pathmaster_ori_matrixA = optarg;
            man_a = 0;
            break;
        case 'b':
            pathmaster_ori_matrixB = optarg;
            man_b = 0;
            break;
        case 'c':
            pathmaster_ori_matrixC = optarg;
            man_c = 0;
            break;
        case '?':
            if (my_rank == 0)
            {

                if (isprint(optopt))
                {

                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                }
                else
                {
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                }
            }
        default:
            printf("Error. Can't process input.\n");
            abort();
        }
    int res = man_a + man_b + man_c;
    if (res != 0)
    {
        if (my_rank == 0)
        {

            printf("\n\n");
            printf("Error. Mismatched number of parameters passed to the program.\n");
            h_rootPrintHelp(0);
            printf("\n\n");
        }
        abort();
    }
    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
}

/**
 * @brief Calcs the result of multiplying two nxn-matrixes.
 * 
 * @param matrix_a 
 * @param matrix_b 
 * @param matrix_c The result is calculated in-place.
 * @param size 
 */
void seq_MatrixMulti(double *matrix_a, double *matrix_b, double *matrix_c, int size)
{
    int i, j, k;
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            for (k = 0; k < size; k++)
                matrix_c[i * size + j] += matrix_a[i * size + k] * matrix_b[k * size + j];
}

void MatrixMatrixMultiply(int n, double *a, double *b, double *c,
                          MPI_Comm comm)
{
    int i;
    int nlocal;
    int npes, dims[2], periods[2];
    int myrank, my2drank, mycoords[2];
    int uprank, downrank, leftrank, rightrank, coords[2];
    int shiftsource, shiftdest;
    MPI_Status status;
    MPI_Comm comm_2d;

    /* Get the communicator related information */
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);

    /* Set up the Cartesian topology */
    dims[0] = dims[1] = sqrt(npes);

    /* Set the periods for wraparound connections */
    periods[0] = periods[1] = 1;

    /* Create the Cartesian topology, with rank reordering */
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

    /* Get the rank and coordinates with respect to the new topology */
    MPI_Comm_rank(comm_2d, &my2drank);
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

    /* Compute ranks of the up and left shifts */
    MPI_Cart_shift(comm_2d, 0, -1, &rightrank, &leftrank);
    MPI_Cart_shift(comm_2d, 1, -1, &downrank, &uprank);

    /* Determine the dimension of the local matrix block */
    nlocal = n / dims[0];

    /* Perform the initial matrix alignment. First for A and then for B */
    MPI_Cart_shift(comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_DOUBLE, shiftdest,
                         1, shiftsource, 1, comm_2d, &status);

    MPI_Cart_shift(comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_DOUBLE,
                         shiftdest, 1, shiftsource, 1, comm_2d, &status);

    /* Get into the main computation loop */
    for (i = 0; i < dims[0]; i++)
    {
        printf("Main loop %d", i);
        seq_MatrixMulti(a, b, c, nlocal); /*c=c+a*b*/

        /* Shift matrix a left by one */
        MPI_Sendrecv_replace(a, nlocal * nlocal, MPI_DOUBLE,
                             leftrank, 1, rightrank, 1, comm_2d, &status);

        /* Shift matrix b up by one */
        MPI_Sendrecv_replace(b, nlocal * nlocal, MPI_DOUBLE,
                             uprank, 1, downrank, 1, comm_2d, &status);
    }
    printf("Finish %d", my_rank);
}

/* GARBAGE
Writes ma
 if (my_rank == 0)
    {
        MPI_File mpi_file;
        int times4 = 4 * 4;
        int times8 = 8 * 8;

        double *master_ori_matrixA4 = malloc(sizeof(double) * times4);
        double *master_ori_matrixA8 = malloc(sizeof(double) * times8);
        double *master_ori_matrixC4 = malloc(sizeof(double) * times4);
        double *master_ori_matrixC8 = malloc(sizeof(double) * times8);
        for (i = 0; i < times4; i++)
            master_ori_matrixA4[i] = i % 4;
        for (i = 0; i < times8; i++)
            master_ori_matrixA8[i] = i % 8;

        for (i = 0; i < times4; i++)
        {
            if (i % 4 == 0)
                printf("\n");
            printf("%f ", master_ori_matrixA4[i]);
        }

        char *pathToResultFile = "./a4x4"; //PATH where to save result
        err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        if (err)
            printf("Error opening the file. \n");
        MPI_File_write(mpi_file, master_ori_matrixA4, times4, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);
        pathToResultFile = "./a8x8"; //PATH where to save result
        err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        if (err)
            printf("Error opening the file. \n");
        MPI_File_write(mpi_file, master_ori_matrixA8, times8, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);
        printf("EASY\n");

        pathToResultFile = "./a4x4"; //PATH where to save result
        double *reload_PicMatrix = malloc(sizeof(double) * times4);
        MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        MPI_File_read(mpi_file, reload_PicMatrix, times4, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);

        printf("R\n");
        for (i = 0; i < times4; i++)
        {
            if (i % 4 == 0)
                printf("\n");
            printf("%f ", reload_PicMatrix[i]);
        }
    }
*/
