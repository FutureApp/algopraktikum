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

int my_rank, world_size; //MPI-STUFF
void *printVector(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint);
void *printVectorNoBar(char tag, double *vector, int dimOfVec, int row, int yourRank, int rankToPrint);
void *printVectorcharNoBar(char tag, unsigned char *vector, int dimOfVec, int row, int yourRank, int rankToPrint);
void *shiftyMatrix(double *matrix_new, double *matrix_old, int sizeOfMatrix)
{
    for (int s = 0; s < sizeOfMatrix; s++)
        matrix_new[s] = matrix_old[s];
}

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
    //printVectorcharNoBar('O', ori_PicMatrix, elemsToHandle, 16, my_rank, 0);

    int sizeOfFilterMatrices = SIZE_D;
    double blurMatrix[SIZE_D] = {0, 0, 1, 0, 0,
                                 0, 2, 4, 2, 0,
                                 1, 4, 9, 4, 1,
                                 0, 2, 4, 2, 0,
                                 0, 0, 1, 0, 0};

    printVectorNoBar('T', blurMatrix, sizeOfFilterMatrices, 5, my_rank, 0);
    for (i = 0; i < sizeOfFilterMatrices; i++)
        blurMatrix[i] = blurMatrix[i] / 37;
    printVectorNoBar('T', blurMatrix, sizeOfFilterMatrices, 5, my_rank, 0);

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

    printVectorNoBar('F', filterMatrix, sizeOfFilterMatrices, 5, my_rank, 0);

    int randPixs = 5;
    int picWidthBIG = picWidth + 2 * randPixs;
    int picHeightBIG = picHeight + 2 * randPixs;
    int sizeOfWorkingMatrix = picWidthBIG * picHeightBIG;
    unsigned char *init_matrixBig = malloc(sizeof(unsigned char) * sizeOfWorkingMatrix);
    unsigned char *new_matrixBig = malloc(sizeof(unsigned char) * sizeOfWorkingMatrix);
    for (i = 0; i < picWidthBIG * picHeightBIG; i++)
    {
        init_matrixBig[i] = 0;
        new_matrixBig[i] = 0;
    }
    //printVectorcharNoBar('w', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);

    int start = randPixs;
    int x, y, v, u;
    int posInOriMatrix = 0;
    int k = 2;
    //printf("BIG B [%d,%d]\n", randPixs, picWidthBIG - randPixs);
    for (y = start; y < (picHeightBIG - randPixs); y++)
        for (x = start; x < (picWidthBIG - randPixs); x++)
        {
            //      printf("working on %d\n", (x + picWidthBIG * i));
            init_matrixBig[x + picWidthBIG * y] = ori_PicMatrix[posInOriMatrix];
            posInOriMatrix++;
        }
    posInOriMatrix = 0;
    int debCounter = 0;

    unsigned char *packLeftBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    unsigned char *packLeftBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    int blockOfElmNumber = picWidth / world_size;

    for (i = 0; i < picHeightBIG; i++)
    {
        int indexPackL0 = my_rank * blockOfElmNumber;
        int indexPackL1 = (my_rank * blockOfElmNumber) + 1;

        int indexPackR0 = ((my_rank + 1) * blockOfElmNumber) - 2;
        int indexPackR1 = ((my_rank + 1) * blockOfElmNumber) - 1;

        int elm00 = init_matrixBig[(i * picWidthBIG) + indexPackL0 + randPixs];
        int elm01 = init_matrixBig[(i * picWidthBIG) + indexPackL1 + randPixs];
        int elm02 = init_matrixBig[(i * picWidthBIG) + indexPackR0 + randPixs];
        int elm03 = init_matrixBig[(i * picWidthBIG) + indexPackR1 + randPixs];

        packLeftBlockToSend[i * 2] = elm00;
        packLeftBlockToSend[(i * 2) + 1] = elm01;

        packRightBlockToSend[i * 2] = elm02;
        packRightBlockToSend[(i * 2) + 1] = elm03;
    }
    printf("[%d] Packing finished\n", my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printVectorcharNoBar('I', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    if (my_rank == 3)
    {
        printf("Left\n");
        for (i = 0; i < picHeightBIG * 2; i++)
        {
            if (i % 2 == 0)
                printf("\n---\n (%d)", i / 2);
            printf("%u ", packLeftBlockToSend[i]);
        }
        printf("Right\n");
        for (i = 0; i < picHeightBIG * 2; i++)
        {
            if (i % 2 == 0)
                printf("\n---\n (%d)", i / 2);
            printf("%u ", packRightBlockToSend[i]);
        }
    }

    abort();
    unsigned char *manipulated_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);
    printVectorcharNoBar('I', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);
    for (y = start; y < (picHeightBIG - randPixs); y++)
        for (x = start; x < (picWidthBIG - randPixs); x++)
        {
            int localSum = 0;
            for (v = 0; v <= 4; v++)
            {
                for (u = 0; u <= 4; u++)
                {
                    double elemOfMatrix = 0;
                    double elemOfFilter = 0;
                    int matrixX = (x + u - k);
                    int matrixYY = y + v - k;
                    int matrixY = (picWidthBIG * y) + picWidthBIG * (v - k);

                    int posInMatrix = matrixX + matrixY;
                    elemOfMatrix = init_matrixBig[posInMatrix];

                    int posInFilter = u + v * 5;
                    elemOfFilter = filterMatrix[posInFilter];
                    localSum += elemOfFilter * elemOfMatrix;

                    //printf("Pos %d x %d I(%d) in Matrix: %3.3f\n", matrixYY, matrixX, posInMatrix, posInMatrix, elemOfMatrix);
                    //              printf("POS(%d) Filter elem: %f | Martix elem: %f\n", posInFilter, elemOfFilter, elemOfMatrix);
                }
                //          printf("\n");
            }
            if (localSum < 0)
                localSum = 0;
            else if (localSum > 255)
                localSum = 255;
            else
                ;
            new_matrixBig[x + y * picWidthBIG] = localSum;
            manipulated_PicMatrix[debCounter] = localSum;
            //printf("localSum =%d\n", localSum);
            debCounter++;
            //for (i = 0; i < 10000000 * 2; i++)  ;
            printVectorcharNoBar('N', new_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);
            //printf("\n\n\n\n\n");
            //printf("\n\n\n\n\n");
            //abort();
        }

    // ------------------------------------------------------------[ RESULT ]--

    if (my_rank == 0)
        printf("------------------------------------------[ Result ]\n");
    //printVectorcharNoBar('r', manipulated_PicMatrix, elemsToHandle, picWidth, my_rank, 0);

    // -------------------------------------------------------[ RESULT SAVE ]--
    // write and reload result.
    if (my_rank == 0)
    {
        char *pathToResultFile = "./result.gray"; //PATH where to save result
        err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fhandle);
        if (err)
            printf("Error opening the file. \n");
        MPI_File_write(fhandle, manipulated_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);
        printf("Result saved. Check < %s >.\n", pathToResultFile);
        unsigned char *reload_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);
        MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fhandle);
        MPI_File_read(fhandle, reload_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);
        printf("------------------------------------------[Result rel.]\n");
        // printVectorcharNoBar('R', reload_PicMatrix, elemsToHandle, picWidth, my_rank, 0);
    }

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
    int index = 0;
    if (yourRank == rankToPrint)
    {
        printf("[%d][%c]  > ", yourRank, tag);
        printf("\n          ");
        for (m = 0; m < hit; m++)
        {
            printf("%3u ", m);
        }
        for (m = 0; m < dimOfVec; m++)
        {
            if ((m % hit) == 0)
            {
                printf("\n%3d       ", index);
                index++;
            }
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
        printf("*Parameter -s <path to picture>   :    Path to picture.\n");
        printf("*Parameter -f <number as Integer> :    Specifies the filter-matrix to apply. Available filters:  \n");
        printf("                                       [0] Blur  \n");
        printf("                                       [1] Sharpen  \n");
        printf("                                       [2] Relief  \n");
        printf("                                       [3] Edge dec.  \n");
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
    int man_f = -1;

    opterr = 0;
    while ((c = getopt(argc, argv, "hs:f:")) != -1)
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
    int res = man_s + man_f;
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