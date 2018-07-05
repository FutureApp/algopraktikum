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
    }
}

void mutex()
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000);
}

int my_rank, world_size; //MPI-STUFF
char *pathMatrixA;
char *pathMatrixB;

int err, i;
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

    err = MPI_File_open(MPI_COMM_SELF, pathMatrixA, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileA);
    MPI_File_get_size(mpi_fileA, &fsizeA);
    err = MPI_File_open(MPI_COMM_SELF, pathMatrixB, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileB);
    MPI_File_get_size(mpi_fileB, &fsizeB);

    elemsToHandleA = fsizeA / (sizeof(double));
    elemsToHandleB = fsizeB / (sizeof(double));
    if (my_rank == 0)
        printf("Elms per row <%d>\n", elemsToHandleA);

    double *matrixA = malloc(sizeof(double) * elemsToHandleA);
    double *matrixB = malloc(sizeof(double) * elemsToHandleB);
    double *matrixC = malloc(sizeof(double) * elemsToHandleB);

    int matrixDim = (int)sqrt(elemsToHandleA);
    MPI_File_read(mpi_fileA, matrixA, elemsToHandleA, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(mpi_fileB, matrixB, elemsToHandleB, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_fileA);
    MPI_File_close(&mpi_fileB);

    mutex();
    h_printQuaMatrixOfDouble('A', matrixA, matrixDim, my_rank, 1);
    mutex();

    if (my_rank == 0)
        printf("Elms per row <%d>\n", matrixDim);
    // -------------------------------------------------------[ CODE ]--

    // -------------------------------------------------------[ CODE ]--

    //###############
    // root reads 2 paths  (A&B matrix) ------------------------------------X
    // root listen to start and quit.
    //###############

    //###############
    // Calculate number of nodes and starts the slaves
    //###############

    //###############
    // Init infrastructure and distribute the values
    // MPI_Cart_create()
    // Each process gets m^2 values where m = n/ srt(p)
    // MPI_Sendrecv_replace and MPI_Card_shift - HOW communication works.
    //###############

    //###############
    // All workers writing their values to file
    // Processing stops if masters get quit()
    // Listening if everything is done works by test!
    //###############

    // -------------------------------------------------------[ RESULT SAVE ]--
    seq_MatrixMulti(matrixA, matrixB, matrixC, matrixDim);

    mutex();
    printf("\n");
    mutex();
    h_printQuaMatrixOfDouble('R', matrixC, matrixDim, my_rank, 0);
    mutex();
    MatrixMatrixMultiply(16, matrixA, matrixB, matrixC, MPI_COMM_WORLD);
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

    opterr = 0;
    while ((c = getopt(argc, argv, "ha:b:")) != -1)
        switch (c)
        {
        case 'h':
            h_rootPrintHelp(my_rank);
            exit(0);
            break;
        case 'a':
            pathMatrixA = optarg;
            man_a = 0;
            break;
        case 'b':
            pathMatrixB = optarg;
            man_b = 0;
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
    int res = man_a + man_b;
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