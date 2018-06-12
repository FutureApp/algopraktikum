
#include <stdio.h>
#include "mpi.h"
#include <math.h>

#include <float.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

void h_rootPrintHelp(int my_rank);
void h_rootPrintMes(int my_rank, char *mes);
void h_setAndCheckParams(int argc, char *argv[]);

char *pathToMatrix; // IN - -pm
char *pathToVector; // IN - -pv
double eps;         // IN - -eps

int my_rank, world_size; //MPI-STUFF

double distanceV(double xOld[], double xNew[], int numberOfCols);
double calcDif(double xOld[], double xNew[], int numberOfCols);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // checks and sets the parameter.
    h_setAndCheckParams(argc, argv);
    //*****

    MPI_File fhandle;
    MPI_Offset fsize;

    int err = 0;
    err = MPI_File_open(MPI_COMM_WORLD, pathToMatrix, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
    err = MPI_File_get_size(fhandle, &fsize);

    if (fsize == 0)
        h_rootPrintMes(my_rank, "Nothing to do. Size of file is 0.\n");

    int dimOfMatrix = sqrt(fsize / sizeof(double));
    int blocksToHandle = dimOfMatrix / world_size;
    int elemsToHandle = dimOfMatrix * blocksToHandle;
    int pushIndexByRank = my_rank * blocksToHandle;
    //xprintf("dimOfMatrix (%d) blocksToHandle (%d) indexToPushBy(%d) elemsToHandle(%d)\n", dimOfMatrix, blocksToHandle, pushIndexByRank, elemsToHandle);
    // Matrix  -- LOADED

    double bufferMatrix[elemsToHandle];
    MPI_File_read_ordered(fhandle, bufferMatrix, elemsToHandle, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);
    //xprintf("HARD MATRIX ENTRY  (%f)\n", bufferMatrix[0]);

    // Vector B -- LOADED
    double bufferVector[dimOfMatrix];
    MPI_File_open(MPI_COMM_WORLD, pathToVector, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
    MPI_File_read(fhandle, bufferVector, dimOfMatrix, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);

    // Vector X -- CREATED
    double xVector[dimOfMatrix];
    for (int i = 0; i < dimOfMatrix; i++)
        xVector[i] = 1;

    double matrixs[blocksToHandle][dimOfMatrix];
    int i = 0;
    for (int row = 0; row < blocksToHandle; ++row)
    {
        for (int col = 0; col < dimOfMatrix; ++col)
        {
            matrixs[row][col] = bufferMatrix[i];
            i++;
        }
    }

    /*
    */
    int isLocalPartDDM = 1;
    for (int xx = 0; xx < world_size; xx++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
        if (my_rank == xx)
        {
            for (int x = 0; x < blocksToHandle; x++)
            {
                //xprintf("ROW [%d] ", x);
                for (int y = 0; y < dimOfMatrix; y++)
                {
                    //xprintf("%f ", matrixs[x][y]);
                }
                //xprintf("\n");
            }

            //check if dominant START
            for (int i = 0; i < blocksToHandle; i++)
            {
                int pushedIBy = i + pushIndexByRank;
                // for each column, finding sum of each row.
                int sum = 0;
                for (int j = 0; j < dimOfMatrix; j++)
                    sum += abs(matrixs[i][j]);

                // removing the diagonal element.
                sum -= abs(matrixs[i][pushedIBy]);
                // checking if diagonal element is less
                // than sum of non-diagonal element.
                if (abs(matrixs[i][pushedIBy]) < sum)
                    isLocalPartDDM = 0;
                //xprintf("[Row of interest %d: Value (%f)]\n", i, matrixs[i][pushedIBy]);
            }
            if (isLocalPartDDM == 0)
            {
                //xprintf("[Node %d]Matrix is not dominant.\n", my_rank);
            }
            else
            {
                //xprintf("[Node %d]Matrix is dominant.\n", my_rank);
            }
        }
        //check if dominant END
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }

    // checks if every part of the matrix fulfill the dd-requirement.
    int worldIsMatrixDD = 0;
    MPI_Allreduce(&isLocalPartDDM, &worldIsMatrixDD, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldIsMatrixDD != world_size)
    {
        if (my_rank == 0)
        {
        }
        //xprintf("Some parts of the matrix doesn't fulfill the DD requirement.\nThe calculation process will stop.");
    }
    if (isLocalPartDDM == 0)
    {
        //xprintf("[Node %d] My part of the matrix doesn't fulfill the DD-requirement.\n");
        exit(1);
    }
    //xprintf("\n\n");
    //xprintf("Jacobi calc. starts.\n");

    // -----------------------------------------------------[ Jacobi - Part ]--

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    int iterCounter = 0;
    int loopFlag = 1;
    while (loopFlag == 1)
    {

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(500);
        int stopAtThisNumber = 60;
        int rankToPrint = 0;
        if (my_rank == rankToPrint)
            printf("\n----------------------------------------- Iteration (%d)--\n", iterCounter);

        if (my_rank == rankToPrint)
        {
            printf("VecX IN  \n");
            for (int x = 0; x < dimOfMatrix; x++)
            {
                printf("%8f ", xVector[x]);
            }
            printf("\n");
        }
        double tempX[dimOfMatrix];
        for (int x = 0; x < dimOfMatrix; x++)
            tempX[x] = 0;

        for (int i = 0; i < blocksToHandle; i++)
        {
            int myIndex = i + pushIndexByRank;

            double entry_a = matrixs[i][myIndex];
            double entry_b = bufferVector[myIndex];

            double firstSum = 0;
            for (int a = 0; a < myIndex; a++)
            {
                firstSum += matrixs[i][a] * xVector[a];
            }
            double secondSum = 0;
            for (int a = (myIndex + 1); a < dimOfMatrix; a++)
            {
                secondSum += matrixs[i][a] * xVector[a];
            }
            double valueOfBrace = entry_b - firstSum - secondSum;
            double componentValue = (1 / entry_a) * valueOfBrace;
            tempX[myIndex] = componentValue;
        }
        if (my_rank == rankToPrint)
        {
            printf("TempVecX END \n");
            for (int x = 0; x < dimOfMatrix; x++)
            {
                printf("%8f ", tempX[x]);
            }
            printf("\n");
        }

        double collectorVec[dimOfMatrix];
        MPI_Allreduce(&tempX, &collectorVec, dimOfMatrix, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (my_rank == rankToPrint)
        {
            printf("CollectorVec END \n");
            for (int x = 0; x < dimOfMatrix; x++)
            {
                printf("%8f ", collectorVec[x]);
            }
            printf("\n");
        }

        double difVectors_local = 99;
        difVectors_local = calcDif(xVector, collectorVec, dimOfMatrix);
        for (int elemIter = 0; elemIter < dimOfMatrix; elemIter++)
        {
            xVector[elemIter] = collectorVec[elemIter];
            collectorVec[elemIter] = 0;
        }
        if (my_rank == rankToPrint)
        {
            printf("xVector END \n");
            for (int x = 0; x < dimOfMatrix; x++)
            {
                printf("%8f ", xVector[x]);
            }
            printf("\n");
        }
        //xprintf("\n");
        if (difVectors_local < eps)
            loopFlag = 0;

        //xprintf("Iteration done (%d)\n", iterCounter);
        //xprintf("#\n");
        //xprintf("#\n");
        //xprintf("#\n");
        //xprintf("#\n");
        if (iterCounter >= 15000000)
        {
            //xprintf("ERROR, aborting calculations\n", iterCounter);
            loopFlag = 0;
        }
        iterCounter++;
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(500);
        if (iterCounter == stopAtThisNumber)
            exit(1);
        /*for (int xx = 0; xx < world_size; xx++)
                        { //--big
                            if (my_rank == xx)
                            { //--inner
                            } //--inner
                        }     //--big
                        */
    }

    //int isFinish = calcDif(0.01, bufferVector, xVector, dimOfMatrix);
    ////xprintf("Calculation is finished (%d)\n", isFinish);
    //print output. *vector print should be easy

    MPI_Finalize(); // finalizing MPI interface
}

double calcDif(double xOld[], double xNew[], int numberOfCols)
{
    //xprintf("Compareing vectors\n");
    //xprintf("Vector 01 = ");
    for (int i = 0; i < numberOfCols; i++)
    {
        //xprintf("%f ", xOld[i]);
    }
    //xprintf("\n");
    //xprintf("Vector 01 = ");
    for (int i = 0; i < numberOfCols; i++)
    {
        //xprintf("%f ", xNew[i]);
    }
    //xprintf("\n");
    int isFinish = 0;
    double dis = 0;
    dis = distanceV(xOld, xNew, numberOfCols);
    int Digs = DECIMAL_DIG;
    //xprintf("[node (%d)] -- distance (%.*e)\n", my_rank, Digs, dis);

    return dis;
}

double distanceV(double xOld[], double xNew[], int numberOfCols)
{
    double sum = 0;
    for (int i = 0; i < numberOfCols; i++)
    {
        sum = sum + (xOld[i] - xNew[i]) * (xOld[i] - xNew[i]);
    }
    return sqrt(sum);
}

/**
 * @brief 
 * Checks if a matrix is dominant.
 * 
 * source: 
 * https://www.geeksforgeeks.org/diagonally-dominant-matrix/
 * 
 * @param m  Matrix m.
 * @param n Dimension of matrix.
 * @return int  1- if dominant
int isLocalPartDDM(double m[i][j], int n)
{
    // for each row
    for (int i = 0; i < n; i++)
    {

        // for each column, finding sum of each row.
        int sum = 0;
        for (int j = 0; j < n; j++)
            sum += abs(m[i][j]);

        // removing the diagonal element.
        sum -= abs(m[i][i]);

        // checking if diagonal element is less
        // than sum of non-diagonal element.
        if (abs(m[i][i]) < sum)
            return 0;
    }
    return 1;
}
 */

/**
 * @brief Prints the help message. Only root (rank=0) will print the message.
 * 
 * @param my_rank  Rank of node. 
 */
void h_rootPrintHelp(int my_rank)
{
    if (my_rank == 0)
    {
        //xprintf("------------------------------------[ HELP ]\n");
        //xprintf("*Parameter -m <path to matrix>   :    Path to file containing the matrix-entrys.\n");
        //xprintf("*Parameter -v <path to vector>   :    Path to file containing the vector-entrys.\n");
        //xprintf("*Parameter -e <number as double>:     Specifies epsilon. Need to be double. \n");
        //xprintf("\n");
        //xprintf("Example call:\n");
        //xprintf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
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

    opterr = 0;
    while ((c = getopt(argc, argv, "hm:v:e:")) != -1)
        switch (c)
        {
        case 'h':
            h_rootPrintHelp(my_rank);
            exit(0);
            break;
        case 'm':
            pathToMatrix = optarg;
            break;
        case 'v':
            pathToVector = optarg;
            break;
        case 'e':
            sscanf(optarg, "%lf", &eps);
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
            abort();
        }

    printf("INPUT(%dr): m = %s, v = %s, eps = %f\n",
           my_rank, pathToMatrix, pathToVector, eps);

    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
}
void h_rootPrintMes(int my_rank, char *mes)
{
    if (my_rank == 0)
    {
        //xprintf("%s\n", mes);
    }
}

/* Prints vector 
    for (int i = 0; i < world_size; i++)
    {
        if (my_rank == i)
        {
            //xprintf("#%dr#\n", my_rank);
            for (int x = 0; x < dimOfMatrix; x++)
            {
                if ((x % dimOfMatrix == 0) && (x != 0))
                {
                    //xprintf("|\n");
                    //xprintf("%f ", bufferVector[x]);
                }
                else
                {
                    //xprintf("%f ", bufferVector[x]);
                }
            }
            //xprintf("|\n");
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
            //xprintf("#%dr#\n", my_rank);
            for (int x = 0; x < elemsToHandle; x++)
            {
                if ((x % dimOfMatrix == 0) && (x != 0))
                {
                    //xprintf("|\n");
                    //xprintf("%f ", bufferMatrix[x]);
                }
                else
                {
                    //xprintf("%f ", bufferMatrix[x]);
                }
            }
            //xprintf("|\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    */

/*Prints B
    //xprintf("Vector B\n");
    for (int i = 0; i < dimOfMatrix; i++)
        //xprintf("%f ", bufferVector[i]);
    //xprintf("\n\n");
    * /

/*Prints X
    //xprintf("Vector X\n");
    for (int i = 0; i < dimOfMatrix; i++)
        //xprintf("%f ", xVector[i]);
    //xprintf("\n\n");
*/