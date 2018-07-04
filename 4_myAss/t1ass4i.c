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

    MPI_File mpi_file;
    MPI_Offset fsize;

    int i;
    int err = 0;

    int picHeight = 0;
    //picWidt is given fromm outside.

    double rowToHandle;
    // checks and sets the parameter.
    h_setAndCheckParams(argc, argv);
    if (my_rank == 0)
    {
        printf("[%d] Program input: (filter:%f)(width:%d)", numberOfFilterTo, picWidth);
    }

    err = MPI_File_open(MPI_COMM_SELF, pathToSrcPic, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
    err = MPI_File_get_size(mpi_file, &fsize);

    picHeight = (fsize / (sizeof(unsigned char)) / picWidth);
    int elemsToHandle = picHeight * picWidth;
    unsigned char *ori_PicMatrix = malloc(sizeof(unsigned char) * elemsToHandle);
    //MPI_File_read(mpi_file, ori_PicMatrix, elemsToHandle, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    //***************************************
    MPI_Datatype col, vector2;
    //MPI_Type_vector(count =1,blocklength = 2, stride = 5, old_type = MPI_INT, &newtype);
    MPI_Type_vector(picHeight, (picWidth / world_size), picWidth, MPI_UNSIGNED_CHAR, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, (picWidth / world_size) * sizeof(unsigned char), &vector2);
    //MPI_File_set_view(mpi_file, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);
    MPI_Type_commit(&vector2);

    MPI_File_read(mpi_file, ori_PicMatrix, picHeight * picWidth, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);
    if (my_rank == 0)
        for (i = 0; i < picWidth * picHeight; i++)
        {
            if (i % picWidth == 0)
                printf("\n");
            printf("%3u ", ori_PicMatrix[i]);
        }
    if (my_rank == 0)
        printf("\n");
    int numberOfElements = (picWidth / world_size) * picHeight;
    int maxNumberOfElement = picWidth * picHeight;
    if (my_rank == 0)
        printf("%d To handle\n", numberOfElements);

    unsigned char *buf_ori_PicMatrix = malloc(sizeof(unsigned char) * numberOfElements);
    for (i = 0; i < numberOfElements; i++)
        buf_ori_PicMatrix[i] = 0;

    MPI_Scatter(ori_PicMatrix, 1, vector2, buf_ori_PicMatrix, numberOfElements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 3)
    {
        printf("\n");
        printf("After Scatter   \n");
        for (i = 0; i < numberOfElements; i++)
        {
            if (i % (picWidth / world_size) == 0)
                printf("\n");
            printf("%3u ", buf_ori_PicMatrix[i]);
        }
    }
    //***************
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
    // push the ori entry to their right position

    int blockOfElmNumber = picWidth / world_size;
    int myBoundL = start + (my_rank * blockOfElmNumber);
    int myBoundR = start + (((my_rank + 1) * blockOfElmNumber) - 1);

    for (y = start; y < (picHeightBIG - randPixs); y++)
        for (x = myBoundL; x <= myBoundR; x++)
        {
            //      printf("working on %d\n", (x + picWidthBIG * i));
            init_matrixBig[x + picWidthBIG * y] = buf_ori_PicMatrix[posInOriMatrix];
            posInOriMatrix++;
        }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    posInOriMatrix = 0;
    printVectorcharNoBar('D', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);

    unsigned char *packLeftBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packRightBlockToSend = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    unsigned char *packRightBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *packLeftBlockToRecv = malloc(sizeof(unsigned char) * picHeightBIG * 2);
    unsigned char *zeros = malloc(sizeof(unsigned char) * picHeightBIG * 2);

    for (i = 0; i < picHeightBIG * 2; i++)
    {
        packLeftBlockToRecv[i] = 0;  //my_rank;
        packRightBlockToRecv[i] = 0; //my_rank;
        zeros[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);

    for (i = 0; i < picHeightBIG; i++)
    {
        int indexPackL0 = my_rank * blockOfElmNumber;
        int indexPackL1 = (my_rank * blockOfElmNumber) + 1;

        int indexPackR0 = ((my_rank + 1) * blockOfElmNumber) - 2;
        int indexPackR1 = ((my_rank + 1) * blockOfElmNumber) - 1;

        unsigned char elm00 = init_matrixBig[(i * picWidthBIG) + indexPackL0 + randPixs];
        unsigned char elm01 = init_matrixBig[(i * picWidthBIG) + indexPackL1 + randPixs];
        unsigned char elm02 = init_matrixBig[(i * picWidthBIG) + indexPackR0 + randPixs];
        unsigned char elm03 = init_matrixBig[(i * picWidthBIG) + indexPackR1 + randPixs];

        packLeftBlockToSend[i * 2] = elm00;
        packLeftBlockToSend[(i * 2) + 1] = elm01;

        packRightBlockToSend[i * 2] = elm02;
        packRightBlockToSend[(i * 2) + 1] = elm03;
        //printf("[%d]                        0:%u 1:%u 2:%u 3:%u\n", my_rank, elm00, elm01, elm02, elm03);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    if (my_rank == 0)
    {
        printf("[%d]Send Left Neigh\n", my_rank);
        for (i = 0; i < picHeightBIG * 2; i++)
        {
            if (i % 2 == 0)
                printf("\n");
            printf("%u ", packRightBlockToSend[i]);
        }
    }
    printVectorcharNoBar('D', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printf("[%d] Packing finished\n", my_rank);
    printVectorcharNoBar('D', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);

    MPI_Status status;
    MPI_Request ch1; // o <------ x
    MPI_Request ch2; // x ------> o
    MPI_Request ch3; // o ------> x
    MPI_Request ch4; // x ------> o

    int send = my_rank;
    int rev = -1;
    int sizeSend = picHeightBIG * 2;

    if (world_size != 1 && world_size != 0)
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

        MPI_Start(&ch1);
        MPI_Start(&ch2);
        MPI_Start(&ch3);
        MPI_Start(&ch4);
        MPI_Wait(&ch1, &status);
        MPI_Wait(&ch2, &status);
        MPI_Wait(&ch3, &status);
        MPI_Wait(&ch4, &status);
    }
    else
        ;

    printVectorcharNoBar('I', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);

    // every node is printing: left Side send, left Side receive |||| right send, right recv

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(2000);
    int currentVIPPixel = 0;
    for (x = 0; x < world_size; x++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(2000);
        printf("\n[%d /%d]\n", my_rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(2000);
        if (my_rank == x)
        {
            printf("\n[%d]  lSend  |  rSend  | |  lReiv  |  rReiv  |\n", my_rank);
            for (i = 0; i < picHeightBIG * 2; i++)
            {
                if (i % 2 == 0)
                    printf("\n");

                printf("[%d] ", my_rank);
                printf("%3u ", packLeftBlockToSend[i]);
                printf("%3u ", packLeftBlockToSend[i + 1]);
                printf("| ");
                printf("%3u ", packRightBlockToSend[i]);
                printf("%3u ", packRightBlockToSend[i + 1]);
                printf("| ");
                printf("| ");
                printf("%3u ", packLeftBlockToRecv[i]);
                printf("%3u ", packLeftBlockToRecv[i + 1]);
                printf("| ");
                printf("%3u ", packRightBlockToRecv[i]);
                printf("%3u ", packRightBlockToRecv[i + 1]);
                i++;
            }
            printf("\n[%d] +++++++++++++++++++++++++++++++++++++++++\n", my_rank);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(2000);
    }

    if (my_rank == 0)
        printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000);
    printf("[%d] Shareing is finished\n", my_rank);
    int local_result_numsOfElms = ((picWidth * picHeight) / world_size);
    unsigned char *local_result_PicMatrix = malloc(sizeof(unsigned char) * (local_result_numsOfElms));

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printVectorcharNoBar('I', init_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);

    for (i = 0; i < world_size; i++)
    {
        if (i == my_rank)
        {
            for (y = start; y < (picHeightBIG - randPixs); y++)
            {
                for (x = myBoundL; x <= myBoundR; x++)
                {
                    int localSum = 0;
                    if (my_rank == 1)
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

                                // take elem from left
                                int easyPosInLeftNBlock = (myBoundL - matrixX) % 2;
                                int posPushByY = easyPosInLeftNBlock + matrixYY * 2;
                                unsigned char elemOfInterest;
                                if (matrixX < myBoundL)
                                {
                                    int easyPosInLeftNBlock = (myBoundL - matrixX) % 2;
                                    printf("====== %d\n", easyPosInLeftNBlock);
                                    int posPushByY = easyPosInLeftNBlock + matrixYY * 2;
                                    elemOfInterest = packLeftBlockToRecv[posPushByY];
                                    printf("[%d] m:%d b:%d YES L  ELMPOS(%u) ELM(%u)\n", my_rank, matrixX, myBoundL, posPushByY, elemOfInterest);
                                }
                                // elm is in working range of node
                                else
                                {
                                    elemOfInterest = init_matrixBig[posInMatrix];
                                    printf("[%d] m:%d b:%d NO L  ELM(%u) \n", my_rank, matrixX, myBoundL, elemOfInterest);
                                }

                                // take elm from right
                                if (matrixX > myBoundR)
                                {
                                    int pos0Or1 = (matrixX - myBoundR) - 1;
                                    elemOfInterest = packRightBlockToRecv[pos0Or1 + posPushByY];
                                    printf("[%d] m:%d b:%d YES R  ELMPOS(%u) ELM(%u)\n", my_rank, matrixX, myBoundR, posPushByY, elemOfInterest);
                                }
                                // elm is in working range of node
                                else
                                {
                                    elemOfInterest = init_matrixBig[posInMatrix];
                                    printf("[%d] m:%d b:%d NO R  ELM(%u)\n", my_rank, matrixX, myBoundR, elemOfInterest);
                                }

                                elemOfMatrix = elemOfInterest;

                                int posInFilter = u + v * 5;
                                elemOfFilter = filterMatrix[posInFilter];
                                localSum += elemOfFilter * elemOfMatrix;

                                //printf("Pos %d x %d I(%d) in Matrix: %3.3f\n", matrixYY, matrixX, posInMatrix, posInMatrix, elemOfMatrix);
                                //              printf("POS(%d) Filter elem: %f | Martix elem: %f\n", posInFilter, elemOfFilter, elemOfMatrix);
                                printf("[%d]+++ u%d\n", my_rank, u);
                            }
                            printf("[%d]################################ u\n", my_rank);
                            printf("[%d]---------------------------------------- v%d\n", my_rank, v);
                        }
                    printf("[%d]############################################## Fv\n", my_rank);
                    if (localSum < 0)
                        localSum = 0;
                    else if (localSum > 255)
                        localSum = 255;
                    else
                        ;

                    printf("[%d]Inserting: (%d)\n", my_rank, localSum);
                    new_matrixBig[x + y * picWidthBIG] = localSum;
                    local_result_PicMatrix[currentVIPPixel] = localSum;
                    //printf("localSum =%d\n", localSum);
                    currentVIPPixel++;
                    //for (i = 0; i < 10000000 * 2; i++)  ;
                    printVectorcharNoBar('N', new_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, 1);

                    //printf("\n\n\n\n\n");
                    //printf("\n\n\n\n\n");
                }
                printf("[%d]############################################## x\n", my_rank);
            }
            printf("[%d]---------------------------------------- y\n", my_rank);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printVectorcharNoBar('A', new_matrixBig, picWidthBIG * picHeightBIG, picWidthBIG, my_rank, world_size - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);

    // ------------------------------------------------------------[ RESULT ]--

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    if (my_rank == 0)
        printf("------------------------------------------[ Result ]\n");

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(2500);
    if (my_rank == world_size - 1)
    {
        printf("[%d] Final print\n", my_rank);
        for (i = 0; i < local_result_numsOfElms; i++)
        {
            if (i % blockOfElmNumber == 0)
                printf("\n");
            printf("%3u ", local_result_PicMatrix[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(2500);

    //for (i = 0; i < local_result_numsOfElms; i++)
    //  local_result_PicMatrix[i] = my_rank;

    if (my_rank == 0)
        for (i = 0; i < picWidth * picHeight; i++)
            ori_PicMatrix[i] = 0;

    MPI_Gather(local_result_PicMatrix, local_result_numsOfElms, MPI_UNSIGNED_CHAR, ori_PicMatrix, 1, vector2, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        for (i = 0; i < picWidth * picHeight; i++)
        {
            if (i == 0)
                printf("After Gather\n");
            if (i % picWidth == 0)
                printf("\n");
            printf("%3u ", ori_PicMatrix[i]);
        }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(2500);

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
        printf("Printing\n          ");

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
            if (m == 81 || m == 85 || m == 89 || m == 93)
                //printf("!!! ");
                printf("%3u ", vector[m]);

            else
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