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

void h_printInputHelp()
{
    printf("\n------------------------------------------------------[INPUT HELP]                             \n");
    printf("Ready for calculation. Please choose one of the following coms to start or abort the calculation:\n");
    printf("To start the calc. enter                             < s >\n");
    printf("To abort the program or an ongoing calcution, enter  < e >\n");
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
        printf("*Parameter -c <path to picture>   :    Path to matrix c.  \n");
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
            pathmaster_1d_matrixA = optarg;
            man_a = 0;
            break;
        case 'b':
            pathmaster_1d_matrixB = optarg;
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

    time_t timestamp_sec; /* timestamp in second */
    time(&timestamp_sec); /* get current time; same as: timestamp_sec = time(NULL)  */
    int result = (float)timestamp_sec;
    char resultFileName[32];
    sprintf(resultFileName, "result_%d.double", result); // puts string into buffer
    printf("%s\n", resultFileName);                      // outputs so you can see it
    printf("-----------------------+++++++++++++++++++++");

    char *ptrC[120];
    //exit(1);
    //ptrA = "./test16x16.double";
    // ptrB = "./test16x16.double";
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
    seq_MatrixMulti(master_1d_matrixA, master_1d_matrixB, master_1d_matrixC, master_matrixDimension);
    if (my_rank == printer)
    {
        printf("A\n");
        for (i = 0; i < elmsOfMatrixA; i++)
        {
            if (i % master_matrixDimension == 0)
            {

                printf("\n");
            }
            printf("%.3f ", master_1d_matrixA[i]);
        }
        printf("\nB\n");
        for (i = 0; i < elmsOfMatrixA; i++)
        {
            if (i % master_matrixDimension == 0)
            {

                printf("\n");
            }
            printf("%.3f ", master_1d_matrixB[i]);
        }
        printf("\nC\n");
        for (i = 0; i < elmsOfMatrixA; i++)
        {
            if (i % master_matrixDimension == 0)
            {

                printf("\n");
            }
            printf("%.3f ", master_1d_matrixC[i]);
        }
    }
    printf("\n\n");

    // ###################################################################################################

    // ----------------------------------------------------------[ Spawn Worker (interComm)] -------------

    int numberOfChilds = master_matrixDimension;
    char *worker_program = "./t2-worker-prog";
    MPI_Comm child;
    int spawnError[numberOfChilds];
    printf("MASTER  spawning childs (%d). This will take a moment.\n", numberOfChilds);
    MPI_Comm_spawn(worker_program, MPI_ARGV_NULL, numberOfChilds, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &child, spawnError);

    // Send info to childs.
    MPI_Bcast(&master_matrixDimension, 1, MPI_INT, MPI_ROOT, child);
    MPI_Bcast(resultFileName, 32, MPI_CHAR, MPI_ROOT, child);
    // ###################################################################################################

    // ---------------------------------------------------------------[ Prepare for Scatter] -------------

    // Convert 1-d matrices to 2-d matrices
    double master_2d_matrixA[master_matrixDimension][master_matrixDimension], master_2d_matrixB[master_matrixDimension][master_matrixDimension];

    for (y = 0; y < master_matrixDimension; y++)
        for (x = 0; x < master_matrixDimension; x++)
        {
            master_2d_matrixA[y][x] = master_1d_matrixA[elmMatCounter];
            master_2d_matrixB[y][x] = master_1d_matrixB[elmMatCounter];
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
    int test = sqrt(numberOfChilds);
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