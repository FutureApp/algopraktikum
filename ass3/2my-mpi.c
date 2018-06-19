
#include <stdio.h>
#include "mpi.h"
#include <math.h>

#include <float.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

void h_rootPrintHelp(int my_rank);
void h_setAndCheckParams(int argc, char *argv[]);

char *pathToMatrix; // IN - -pm
char *pathToVector; // IN - -pv
double eps;         // IN - -eps

int my_rank, world_size; //MPI-STUFF

double distanceV(double xOld[], double xNew[], int numberOfCols);
double calcDif(double xOld[], double xNew[], int numberOfCols);

int main(int argc, char *argv[])
{
    int i;
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // checks and sets the parameter.
    h_setAndCheckParams(argc, argv);
    //*****

    MPI_File fhandle;
    MPI_Offset fsize;

    int err = 0;
    // root loads the data

    int dimOfMatrix = 1;
    int blocksToHandle = 1;
    int elemsToHandle = 1;
    int pushIndexByRank = 0;

    if (my_rank == 0) // --- root part (load) Start
    {
        err = MPI_File_open(MPI_COMM_SELF, pathToMatrix, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
        err = MPI_File_get_size(fhandle, &fsize);

        if (fsize == 0)
        {
            printf("[Node (%d)] File has size of 0. Nothing to do! Calculation will stop.\n", my_rank);
            exit(1);
        }
        dimOfMatrix = sqrt(fsize / sizeof(double));
    }

    MPI_Bcast(&dimOfMatrix, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // root sync elements

    blocksToHandle = dimOfMatrix / world_size;
    double bufferMatrix[dimOfMatrix * dimOfMatrix];
    double bufferVector[dimOfMatrix];

    //zeros all entry's  DEBUG
    for (int i = 0; i < dimOfMatrix * dimOfMatrix; i++)
        bufferMatrix[i] = 0;
    for (int i = 0; i < dimOfMatrix; i++)
        bufferVector[i] = 0;

    for (int i = 0; i < world_size; i++)
    {
        if (my_rank == i)
            printf("[node %d] (Dim=%d)(Blocks=%d).\n", my_rank, dimOfMatrix, blocksToHandle);
    }

    if (my_rank == 0) // --- root part (load) Start
    {
        int blocksToHandle_root = dimOfMatrix;
        // important for later steps (snyc with all nodes);
        elemsToHandle = dimOfMatrix * blocksToHandle_root;
        pushIndexByRank = my_rank * blocksToHandle_root;
        printf("[node %d] dimMatrix(%d) blocksToHandle_root(%d) elemsToHandle(%d) pushIndexByRank(%d)\n", my_rank, dimOfMatrix, blocksToHandle_root, elemsToHandle, pushIndexByRank);

        // Matrix  -- LOAD
        MPI_File_read_ordered(fhandle, bufferMatrix, elemsToHandle, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);

        // Vector B -- LOAD
        MPI_File_open(MPI_COMM_SELF, pathToVector, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
        MPI_File_read(fhandle, bufferVector, dimOfMatrix, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);

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
        // prints root B
        printf("|\n");
        if (my_rank == 0)
        {
            printf("[node %d [vecB01]] ", my_rank);
            for (int i = 0; i < 8; i++)
            {
                printf(" %f ", bufferVector[i]);
            }
            printf("|\n");
        }
        printf("\n");
    } // --- root part (load) END

    if (my_rank == 0)
    {
        printf("[node %d [vecB02]] ", my_rank);
        for (int i = 0; i < 8; i++)
        {
            printf(" %f ", bufferVector[i]);
        }
        printf("\n");
    }
    double *rand_nums;
    int elements_per_proc = ((dimOfMatrix * dimOfMatrix) / world_size);
    if (my_rank == 0)
    {
        rand_nums = bufferMatrix;
    }

    // Create a buffer that will hold a subset of the random numbers
    double *sub_rand_nums = malloc(sizeof(double) * elements_per_proc);

    MPI_Datatype vector1, vector2;

    int linesOfMatrix = dimOfMatrix;
    int columnsOfMatrixForProc = dimOfMatrix / world_size;
    int elemsToHandleEach = columnsOfMatrixForProc * columnsOfMatrixForProc;

    MPI_Type_vector(linesOfMatrix, columnsOfMatrixForProc, linesOfMatrix, MPI_DOUBLE, &vector1);
    MPI_Type_commit(&vector1);
    MPI_Type_create_resized(vector1, 0, columnsOfMatrixForProc * sizeof(double), &vector2);
    MPI_Type_commit(&vector2);

    double *new_matrixBuffer = malloc(sizeof(double) * dimOfMatrix * dimOfMatrix);
    double *new_matrixA = malloc(sizeof(double) * dimOfMatrix * dimOfMatrix);
    new_matrixA = bufferMatrix;
    MPI_Scatter(new_matrixA, 1, vector2, new_matrixBuffer, linesOfMatrix * columnsOfMatrixForProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *new_vectorBuffer = malloc(sizeof(double) * dimOfMatrix);
    double *new_vectorB = malloc(sizeof(double) * dimOfMatrix);

    int numOfElemVec = dimOfMatrix / world_size;
    new_vectorB = bufferVector;
    MPI_Scatter(new_vectorB, numOfElemVec, MPI_DOUBLE, new_vectorBuffer, numOfElemVec, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int x;
    for (x = 0; x < world_size; x++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(500);
        if (my_rank == x)
        {
            printf("[%d Entrys of Matrix]\n", my_rank);
            for (i = 0; i < ((dimOfMatrix * dimOfMatrix) / world_size); i++)
            {
                if (i % (dimOfMatrix / world_size) == 0)
                    printf("---------------\n", my_rank, i, new_matrixBuffer[i]);

                printf("[%d : %d] %f\n", my_rank, i, new_matrixBuffer[i]);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    if (my_rank == 0)
        printf("Matrix Scatterd \n");
    for (i = 0; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100);
        if (my_rank == i)
        {
            for (int x = 0; x < dimOfMatrix; x++)
            {
                if (i % dimOfMatrix == 0)
                    printf("----------------------------- \n", my_rank, *(new_matrixBuffer + x));
                printf("[node %d:%d] Old Scatters (%f) \n", my_rank, x, *(new_matrixBuffer + x));
                printf("[node %d:%d] New Scatters (%f) \n", my_rank, x, new_matrixBuffer[x]);
            }
            printf("[node %d] END \n", my_rank);
        }
    }
    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100);
    if (my_rank == 1)
        printf("Vector Scatterd \n");
    for (i = 0; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100);
        if (my_rank == i)
        {
            printf("[node %d] IN \n", my_rank);
            for (int x = 0; x < numOfElemVec; x++)
            {
                if (i % numOfElemVec == 0)
                    printf("xxxxxxxxxxxxxxxxxx \n", my_rank, *(new_vectorBuffer + x));
                printf("[node %d:%d] After Scatters (%f) \n", my_rank, x, *(new_vectorBuffer + x));
            }
            printf("[node %d] END \n", my_rank);
        }
    }
    double *new_vectorX = malloc(sizeof(double) * dimOfMatrix);
    double *old_vectorX = malloc(sizeof(double) * dimOfMatrix);
    for (i = 0; i < dimOfMatrix; i++)
    {
        new_vectorX[i] = 0;
        old_vectorX[i] = 1;
    }
    // init END

    // new_vectorB; new_matrixA;  new_vectorX; old_vectorX;
    // jacobi-section

    int calcIsFinished = 0;
    int iterationCount = 0;
    double lastEps = -1;
    while (calcIsFinished == 0)
    {
        if (my_rank == 0)
            printf("[%d] ---------------------------- [iteration %d] lastEps(%f)\n", my_rank, iterationCount, lastEps);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100);

        int nodeHoldsA = 0;
        int innerCounter = 0;
        printf("Handle %d", blocksToHandle);
        for (i = 0; i < dimOfMatrix; i++)
        {
            double localSum = 0;
            for (x = 0; x < blocksToHandle; x++)
            {
                localSum += new_matrixBuffer[x + (blocksToHandle * i)];
            }
            // DEBUG
            double rootSum = 0;
            if (my_rank == 0)
            {

                for (x = 0; x < dimOfMatrix; x++)
                {
                    rootSum += new_matrixA[x + (i * dimOfMatrix)];
                }
            }
            // DEBUG END
            if ((i % blocksToHandle) == 0 && (i != 0))
            {
                nodeHoldsA++;
            }
            // Reduce ...
            double world_sum = 0;
            // printf("[%d A %d] localSum = %f worldSum = %f\n", my_rank, localSum, world_sum, i);
            MPI_Reduce(&localSum, &world_sum, 1, MPI_DOUBLE, MPI_SUM, nodeHoldsA, MPI_COMM_WORLD);
            // printf("[%d E %d] localSum = %f worldSum = %f\n", my_rank, localSum, world_sum, i);

            double diaElem = 0;
            double Xki = 0;
            if (my_rank == nodeHoldsA)
            {
                int diaElemPos = (i % blocksToHandle) + (i * blocksToHandle);
                diaElem = new_matrixBuffer[diaElemPos];
                world_sum -= diaElem; // in reduce-step added but wrong.

                int posInB = diaElemPos % blocksToHandle;
                double valueOfB = new_vectorBuffer[posInB];
                Xki = (1 / diaElem) * (valueOfB - world_sum);
                // printf(" [%d DIA at pos %d] %f | Bvalue (%f) | Xki-value (%f)\n", my_rank, diaElemPos, diaElem, new_vectorBuffer[posInB], Xki);
            }

            // MPI_Bcast( void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
            MPI_Bcast(&Xki, 1, MPI_DOUBLE, nodeHoldsA, MPI_COMM_WORLD);
            new_vectorX[i] = Xki;
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(500);

            // node with dia calcs
            // Scatter new X  by node with dia

            MPI_Barrier(MPI_COMM_WORLD);
            usleep(500);
            for (x = 0; x < world_size; x++)
            {
                // if (my_rank == x)
                //   printf("[%d sum(%f) it(%d) \n", my_rank, localSum, i);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            usleep(1000);
            if (my_rank == 0)
            {
                //printf("[%d Rsum(%f) it(%d) aHolder(%d)]]\n", my_rank, rootSum, i, nodeHoldsA);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(1000);
        } // calc newX-vector END

        innerCounter = 0;
        nodeHoldsA = 0;
        // exit(1);
        double curEps = calcDif(old_vectorX, new_vectorX, dimOfMatrix);
        old_vectorX = new_vectorBuffer;
        if (curEps < eps)
            calcIsFinished = 1;
        lastEps = curEps;
        iterationCount++;
    }

    for (int v = 0; v < world_size; v++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(1000);
        if (my_rank == v)
        {
            printf("[%d] ", my_rank);
            for (int m = 0; m < dimOfMatrix; m++)
            {
                printf("%f ", new_vectorX[m]);
            }
            printf("\n");
        }
    }

    printf("[node %d] ExEnd.\n", my_rank);
    MPI_Finalize(); // finalizing MPI interface
}

double calcDif(double xOld[], double xNew[], int numberOfCols)
{
    int isFinish = 0;
    double dis = 0;
    dis = distanceV(xOld, xNew, numberOfCols);
    int Digs = DECIMAL_DIG;
    return dis;
}

double distanceV(double xOld[], double xNew[], int numberOfCols)
{
    double sum = 0;
    for (int i = 0; i < numberOfCols; i++)
        sum = sum + (xOld[i] - xNew[i]) * (xOld[i] - xNew[i]);
    return sqrt(sum);
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