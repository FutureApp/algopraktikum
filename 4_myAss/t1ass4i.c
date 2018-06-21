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

char *pathToSrcPic; // IN - -pm

int my_rank, world_size; //MPI-STUFF
void *printVector(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint);
void *printVectorNoBar(char tag, double *vector, int dimOfVec, int row, int yourRank, int rankToPrint);
void *printVectorcharNoBar(char tag, unsigned char *vector, int dimOfVec, int row, int yourRank, int rankToPrint);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_File fhandle;
    MPI_Offset fsize;

    int i;
    int err = 0;

    int picHeight = 0;
    int picWidth = 0;
    picWidth = 16; // Because 1280 pixels per row.
    double rowToHandle;
    // checks and sets the parameter.
    h_setAndCheckParams(argc, argv);
    //*****

    err = MPI_File_open(MPI_COMM_SELF, pathToSrcPic, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
    err = MPI_File_get_size(fhandle, &fsize);

    picHeight = (fsize / (sizeof(unsigned char)) / picWidth);
    int elemsToHandle = picHeight * picWidth;
    // printf("H %d W %d\n", picHeight, picWidth);
    unsigned char *ori_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);
    MPI_File_read(fhandle, ori_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);
    printVectorcharNoBar('O', ori_PicMatrix, elemsToHandle, 16, my_rank, 0);

    int sizeOfFilterMatrices = 5 * 5;
    double blurMatrix[5 * 5] = {0, 0, 1, 0, 0,
                                0, 2, 4, 2, 0,
                                1, 4, 9, 4, 1,
                                0, 2, 4, 2, 0,
                                0, 0, 1, 0, 0};

    printVectorNoBar('T', blurMatrix, sizeOfFilterMatrices, 5, my_rank, 0);
    for (i = 0; i < sizeOfFilterMatrices; i++)
        blurMatrix[i] = blurMatrix[i] / 37;
    printVectorNoBar('T', blurMatrix, sizeOfFilterMatrices, 5, my_rank, 0);

    double sharpenMatrix[5 * 5] = {0, 0, 0, 0, 0,
                                   0, 0, -1, 0, 0,
                                   0, -1, 5, -1, 0,
                                   0, 0, -1, 0, 0,
                                   0, 0, 0, 0, 0};
    double reliefMatrix[5 * 5] = {0, 0, 0, 0, 0,
                                  0, -2, -1, 0, 0,
                                  0, -1, 1, 1, 0,
                                  0, 0, 1, 2, 0,
                                  0, 0, 0, 0, 0};

    double edgeDMatrix[5 * 5] = {0, 0, 0, 0, 0,
                                 0, 1, 2, 1, 0,
                                 0, 2, -12, 2, 0,
                                 0, 1, 2, 1, 0,
                                 0, 0, 0, 0, 0};
    for (i = 0; i < sizeOfFilterMatrices; i++)
        edgeDMatrix[i] = edgeDMatrix[i] / 4;

    int randPixs = 5;
    int picWithBIG = picWidth + 2 * randPixs;
    int picHeightBIG = picHeight + 2 * randPixs;
    int sizeOfWorkingMatrix = picWithBIG * picHeightBIG;
    unsigned char *new_matrixBIG = malloc(sizeof(unsigned char) * sizeOfWorkingMatrix);
    for (i = 0; i < picWithBIG * picHeightBIG; i++)
        new_matrixBIG[i] = 0;
    //printVectorcharNoBar('w', new_matrixBIG, picWithBIG * picHeightBIG, picWithBIG, my_rank, 0);

    int start = randPixs;
    int x, y, v, u;
    int posInOriMatrix = 0;
    int k = 2;
    //printf("BIG B [%d,%d]\n", randPixs, picWithBIG - randPixs);
    for (y = start; y < (picHeightBIG - randPixs); y++)
        for (x = start; x < (picWithBIG - randPixs); x++)
        {
            //      printf("working on %d\n", (x + picWithBIG * i));
            new_matrixBIG[x + picWithBIG * y] = ori_PicMatrix[posInOriMatrix];
            posInOriMatrix++;
        }
    posInOriMatrix = 0;
    printVectorcharNoBar('I', new_matrixBIG, picWithBIG * picHeightBIG, picWithBIG, my_rank, 0);
    for (y = start; y < (picHeightBIG - randPixs); y++)
        for (x = start; x < (picWithBIG - randPixs); x++)
        {

            for (v = 0; v < 4; v++)
                for (u = 0; u < 4; u++)
                {
                    double elemOfMatrix = 0;
                    double elemOfFilter = 0;
                    int matrixX = (x + u - k);
                    int matrixY = (picWithBIG * y) + v - k;
                    
                    int matrixYY = y + v - k;
                    int posInMatrix = matrixX + matrixY;
                    
                    elemOfMatrix = new_matrixBIG[posInMatrix];
                    printf("Pos %d x %d  in Matrix: %f\n", matrixYY, matrixX, posInMatrix, elemOfMatrix);
                }
            abort();
        }

    printf("[node %d] E %d.\n", my_rank, start);
    unsigned char test = 244;
    double testChar = test;
    testChar--;
    unsigned char testnew = testChar;
    printf("[node %d] %u.\n", my_rank, testnew);
    printf("[node %d] ExEnd.\n", my_rank);
    MPI_Finalize(); // finalizing MPI interface
}

/*
  * @brief Prints a vector with tag-mes. 
  * 
  * @param tag tag to recognize  message.
  * @param vector A vector.
  * @param dimOfVec Dimension of the given vector.
  * @param yourRank Rank of node (my_rank).
  * @param rankToPrint Specs. which node has the privilege to print.
  * @return void*  Nothing.
  */
void *printVector(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint)
{
    if (yourRank == rankToPrint)
    {
        printf("[%d][%c]  > ", yourRank, tag);
        for (int m = 0; m < dimOfVec; m++)
        {
            printf("%f ", vector[m]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000);
}
/**
  * @brief Prints a vector with tag-mes. 
  * 
  * @param tag tag to recognize  message.
  * @param vector A vector.
  * @param dimOfVec Dimension of the given vector.
  * @param yourRank Rank of node (my_rank).
  * @param rankToPrint Specs. which node has the privilege to print.
  * @return void*  Nothing.
  */
void *printVectorNoBar(char tag, double *vector, int dimOfVec, int row, int yourRank, int rankToPrint)
{
    int hit = dimOfVec / row;
    if (yourRank == rankToPrint)
    {
        printf("[%d][%c]  > ", yourRank, tag);
        for (int m = 0; m < dimOfVec; m++)
        {
            if ((m % hit) == 0)
                printf("\n       ");
            printf("%.1f ", vector[m]);
        }
        printf("\n");
    }
}

/**
  * @brief Prints a vector with tag-mes. 
  * 
  * @param tag tag to recognize  message.
  * @param vector A vector.
  * @param dimOfVec Dimension of the given vector.
  * @param yourRank Rank of node (my_rank).
  * @param rankToPrint Specs. which node has the privilege to print.
  * @return void*  Nothing.
  */
void *printVectorcharNoBar(char tag, unsigned char *vector, int dimOfVec, int row, int yourRank, int rankToPrint)
{
    int hit = dimOfVec / row;
    int m = 0;
    if (yourRank == rankToPrint)
    {
        printf("[%d][%c]  > ", yourRank, tag);
        printf("\n       ");
        for (m = 0; m < hit; m++)
        {
            printf("%3u ", m);
        }
        for (m = 0; m < dimOfVec; m++)
        {
            if ((m % hit) == 0)
                printf("\n       ");
            printf("%3u ", vector[m]);
        }
        printf("\n");
    }
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
        printf("*Parameter -m <path to matrix>   :    Path to file containing the matrix-entrys.\n");
        printf("*Parameter -v <path to vector>   :    Path to file containing the vector-entrys.\n");
        printf("*Parameter -e <number as double>:     Specifies epsilon. Need to be double. \n");
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
    int man_s = -1;

    opterr = 0;
    while ((c = getopt(argc, argv, "hs:")) != -1)
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
    int res = man_s;
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

/* Prints vector 
    for (int i = 0; i < world_size; i++)
    {
        if (my_rank == i)
        {
            printf("#%dr#\n", my_rank);
            for (int x = 0; x < dimOfMatrix; x++)
            {
                if ((x % dimOfMatrix == 0) && (x != 0))
                {
                    printf("|\n");
                    printf("%f ", bufferVector[x]);
                }
                else
                {
                    printf("%f ", bufferVector[x]);
                }
            }
            printf("|\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    */
/* Prints matrix 
    for (int i = 0; i < world_size; i++)
    {
        if (my_rank == i)
        {
            printf("#%dr#\n", my_rank);
            for (int x = 0; x < elemsToHandle; x++)
            {
                if ((x % dimOfMatrix == 0) && (x != 0))
                {
                    printf("|\n");
                    printf("%f ", bufferMatrix[x]);
                }
                else
                {
                    printf("%f ", bufferMatrix[x]);
                }
            }
            printf("|\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    */

/*Prints B
    printf("Vector B\n");
    for (int i = 0; i < dimOfMatrix; i++)
        printf("%f ", bufferVector[i]);
    printf("\n\n");
    * /

/*Prints X
    printf("Vector X\n");
    for (int i = 0; i < dimOfMatrix; i++)
        printf("%f ", xVector[i]);
    printf("\n\n");
*/

/* Wrapper
   for (int l = 0; l < world_size; l++)
    {
        if (my_rank == l)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(100);
        }
    }
    */