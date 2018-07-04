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

#define SIZE_E 5
#define SIZE_D SIZE_E *SIZE_E

void h_rootPrintHelp(int my_rank);
void h_setAndCheckParams(int argc, char *argv[]);

char *pathToSrcPic;      // IN - -s
double numberOfFilterTo; // IN - -f
int picWidth;            // IN - -w

int my_rank, world_size; //MPI-STUFF
{
    for (int s = 0; s < sizeOfMatrix; s++)
        matrix_new[s] = matrix_old[s];
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_File mpi_file;
    MPI_Offset fsize;

    int i;
    int err = 0;

    int picHeight = 0;
    double rowToHandle;
    // checks and sets the parameter.
    h_setAndCheckParams(argc, argv);

    //***************
    int sizeOfFilterMatrices = SIZE_D;
    double blurMatrix[SIZE_D] = {0, 0, 1, 0, 0,
                                 0, 2, 4, 2, 0,
                                 1, 4, 9, 4, 1,
                                 0, 2, 4, 2, 0,
                                 0, 0, 1, 0, 0};

    for (i = 0; i < sizeOfFilterMatrices; i++)
        blurMatrix[i] = blurMatrix[i] / 37;

    double sharpenMatrix[SIZE_D] = {0, 0, 0, 0, 0,
                                    0, 0, -1, 0, 0,
                                    0, -1, 5, -1, 0,
                                    0, 0, -1, 0, 0,
                                    0, 0, 0, 0, 0};
    double reliefMatrix[SIZE_D] = {0, 0, 0, 0, 0,
                                   0, -2, -1, 0, 0,
                                   0, -1, 1, 1, 0,
                                   0, 0, 1, 2, 0,
                                   0, 0, 0, 0, 0};

    double edgeDMatrix[SIZE_D] = {0, 0, 0, 0, 0,
                                  0, 1, 2, 1, 0,
                                  0, 2, -12, 2, 0,
                                  0, 1, 2, 1, 0,
                                  0, 0, 0, 0, 0};
    for (i = 0; i < sizeOfFilterMatrices; i++)
        edgeDMatrix[i] = edgeDMatrix[i] / 4;

    int filterToApply = (int)numberOfFilterTo;
    double filterMatrix[SIZE_D];
    switch (filterToApply)
    {
        {
        case 0:
            if (my_rank == 0)
                printf("Applying blur-filter\n");
            shiftyMatrix(filterMatrix, blurMatrix, SIZE_D);
            break;
        case 1:
            if (my_rank == 0)
                printf("Applying sharpen-filter\n");
            shiftyMatrix(filterMatrix, sharpenMatrix, SIZE_D);
            break;
        case 2:
            if (my_rank == 0)
                printf("Applying relief:-filter\n");
            shiftyMatrix(filterMatrix, reliefMatrix, SIZE_D);
            break;
        case 3:
            if (my_rank == 0)
                printf("Applying edge.dec-filter\n");
            shiftyMatrix(filterMatrix, edgeDMatrix, SIZE_D);
            break;
        default:
            printf("Don't know which filter-matrix to apply. Execution will terminate now.");
            h_rootPrintHelp(my_rank);
            abort();
            break;
        }
    }

    if (my_rank == 0)
        printf("[%d] Program input: (filter:%f)(width:%d)", numberOfFilterTo, picWidth);

    // ------------------------------------------------------------[ Read data ]--
    err = MPI_File_open(MPI_COMM_SELF, pathToSrcPic, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
    err = MPI_File_get_size(mpi_file, &fsize);

    picHeight = (fsize / (sizeof(unsigned char)) / picWidth);
    int elemsToHandle = picHeight * picWidth;
    unsigned char *ori_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);

    MPI_Datatype col, vector2;
    MPI_Type_vector(picHeight, (picWidth / world_size), picWidth, MPI_UNSIGNED_CHAR, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, (picWidth / world_size) * sizeof(unsigned char), &vector2);
    MPI_Type_commit(&vector2);

    MPI_File_read(mpi_file, ori_PicMatrix, picHeight * picWidth, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);
    // ------------------------------------------------------------[ Scatter data ]--
    //TODO

    // ------------------------------------------------------------[ Prepare for share ]--
    unsigned char *packLeftBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    unsigned char *packRightBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packLeftBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    // ------------------------------------------------------------[ channel init ]--
    MPI_Status status;
    MPI_Request ch1; // o <------ x
    MPI_Request ch2; // x ------> o
    MPI_Request ch3; // o ------> x
    MPI_Request ch4; // x ------> o

    int sizeSend = picHeightBIG * 2;
    // Here we set the channels for later communication.
    if (1 == 1)
    {
        if (my_rank % 2 == 0)
        {

            MPI_Recv_init(packRightBlockToRecv, sizeSend, MPI_UNSIGNED_CHAR, my_rank + 1, 1, MPI_COMM_WORLD, &ch1);
            MPI_Send_init(packRightBlockToSend, sizeSend, MPI_UNSIGNED_CHAR, my_rank + 1, 1, MPI_COMM_WORLD, &ch2);

            if (my_rank == 0)
            {

                MPI_Send_init(zeros, sizeSend, MPI_UNSIGNED_CHAR, world_size - 1, 1, MPI_COMM_WORLD, &ch3);
                MPI_Send_init(zeros, sizeSend, MPI_UNSIGNED_CHAR, world_size - 1, 1, MPI_COMM_WORLD, &ch4);
            }
            else
            {
                MPI_Send_init(packLeftBlockToSend, sizeSend, MPI_UNSIGNED_CHAR, my_rank - 1, 1, MPI_COMM_WORLD, &ch3);
                MPI_Recv_init(packLeftBlockToRecv, sizeSend, MPI_UNSIGNED_CHAR, my_rank - 1, 1, MPI_COMM_WORLD, &ch4);
            }
        }
        else
        {
            MPI_Send_init(packLeftBlockToSend, sizeSend, MPI_UNSIGNED_CHAR, my_rank - 1, 1, MPI_COMM_WORLD, &ch1);
            MPI_Recv_init(packLeftBlockToRecv, sizeSend, MPI_UNSIGNED_CHAR, my_rank - 1, 1, MPI_COMM_WORLD, &ch2);

            if (my_rank == world_size - 1)
            {
                MPI_Send_init(zeros, sizeSend, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &ch3);
                MPI_Send_init(zeros, sizeSend, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &ch4);
            }
            else
            {
                MPI_Recv_init(packRightBlockToRecv, sizeSend, MPI_UNSIGNED_CHAR, my_rank + 1, 1, MPI_COMM_WORLD, &ch3);
                MPI_Send_init(packRightBlockToSend, sizeSend, MPI_UNSIGNED_CHAR, my_rank + 1, 1, MPI_COMM_WORLD, &ch4);
            }
        }
    }
    // Perpare data for exchange
    // ------------------------------------------------------------[ Exchange data ]--
    MPI_Start(&ch1);
    MPI_Start(&ch2);
    MPI_Start(&ch3);
    MPI_Start(&ch4);
    MPI_Wait(&ch1, &status);
    MPI_Wait(&ch2, &status);
    MPI_Wait(&ch3, &status);
    MPI_Wait(&ch4, &status);

    //

    // ------------------------------------------------------------[ RESULT ]--
    if (my_rank == 0)
        printf("------------------------------------------[ Result ]\n");

    //MPI_Gather(local_result_PicMatrix, local_result_numsOfElms, MPI_UNSIGNED_CHAR, ori_PicMatrix, 1, vector2, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------[ RESULT SAVE ]--
    // write and reload result.
    if (my_rank == 0)
    {
        char *pathToResultFile = "./result.gray"; //PATH where to save result
        err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        if (err)
            printf("Error opening the file. \n");
        MPI_File_write(mpi_file, local_result_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);
        printf("Result saved. Check < %s >.\n", pathToResultFile);
        unsigned char *reload_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);
        MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_file);
        MPI_File_read(mpi_file, reload_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&mpi_file);
        printf("------------------------------------------[Result rel.]\n");
        // printVectorcharNoBar('R', reload_PicMatrix, elemsToHandle, picWidth, my_rank, 0);
    }

    printf("[node %d] ExEnd.\n", my_rank);
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
        printf("*Parameter -s <path to picture>   :    Path to picture.\n");
        printf("*Parameter -f <number as Integer> :    Specifies the filter-matrix to apply. Available filters:  \n");
        printf("                                       [0] Blur  \n");
        printf("                                       [1] Sharpen  \n");
        printf("                                       [2] Relief  \n");
        printf("                                       [3] Edge dec.  \n");
        printf("\n");
        printf("*Parameter -w <number as Integer> :    Specifies the width of the given picture. Option is crucial");
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
    int man_s = -1;
    int man_f = -1;
    int man_w = -1;

    opterr = 0;
    while ((c = getopt(argc, argv, "hs:f:w:")) != -1)
        switch (c)
        {
        case 'h':
            h_rootPrintHelp(my_rank);
            exit(0);
            break;
        case 's':
            pathToSrcPic = optarg;
            man_s = 0;
            break;
        case 'f':
            sscanf(optarg, "%lf", &numberOfFilterTo);
            man_f = 0;
            break;
        case 'w':
            sscanf(optarg, "%d", &picWidth);
            man_w = 0;
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
    int res = man_s + man_f + man_w;
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
    printf("INPUT(%dr): s = %s \n",
           my_rank, pathToSrcPic);

    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
}