
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

    /*
    MPI_Scatter(void *send_data, int send_count, MPI_Datatype send_datatype, void *recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator);
    MPI_Scatter(rand_nums, elements_per_proc, MPI_FLOAT, sub_rand_nums,
                elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(m, elements_per_proc, MPI_DOUBLE, sub_rand_nums,
                elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    */

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1500);
    for (int i = 0; i < world_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
        if (my_rank == i)
        {
            printf("#%dr#\n", my_rank);
            for (int x = 0; x < (dimOfMatrix * dimOfMatrix); x++)
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

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[node %d] ExEnd.\n", my_rank);
    /* Start HIT 
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(500);
    exit(1);

    // Vector X -- CREAT
    double xVector[dimOfMatrix];
    for (int i = 0; i < dimOfMatrix; i++)
        xVector[i] = 1;

    // Creates 2d-matrix.
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

    int isLocalPartDDM = 1;
    for (int xx = 0; xx < world_size; xx++)
    {
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
                printf("[Node %d]Matrix is not dominant.\n", my_rank);
            }
            else
            {
                printf("[Node %d]Matrix is dominant.\n", my_rank);
            }
        }
        //check if dominant END
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }

    // checks if every part of the matrix fulfills the dd-requirement.
    int worldIsMatrixDD = 0;
    MPI_Allreduce(&isLocalPartDDM, &worldIsMatrixDD, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldIsMatrixDD != world_size)
    {
        printf("Some parts of the matrix doesn't fulfill the DD requirement.\nThe calculation process will stop now.");
        exit(1);
    }

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
        if (difVectors_local < eps)
            loopFlag = 0;

        iterCounter++;

        if (iterCounter >= stopAtThisNumber)
        {
            printf("ERROR, aborting calculations\n", iterCounter);
            loopFlag = 0;
            exit(1);
        }
    }

    // -------------------------------------------------------[ Save result ]--
    char *pathToResultFile = "./res";
    err = MPI_File_open(MPI_COMM_WORLD, pathToResultFile, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fhandle);
    if (err)
    {
        MPI_Abort(MPI_COMM_WORLD, 911);
    }
    double me = 10;
    double buf[blocksToHandle];
    for (int index = 0; index < blocksToHandle; index++)
    {
        buf[index] = xVector[index + pushIndexByRank];
        printf("saved %f \n", xVector[index + pushIndexByRank]);
    }

    err = MPI_File_write_ordered(fhandle, &buf, blocksToHandle, MPI_DOUBLE, MPI_STATUS_IGNORE);
    if (err)
    {
        printf("Error writing to file. \n");
    }
    MPI_File_close(&fhandle);

    // ------------------------------------------------------[ print Result ]--
    if (my_rank == 0)
    {
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("----------------------------------------------[ Result ]--\n");
        printf("Result-vector: \n");
        for (int x = 0; x < dimOfMatrix; x++)
        {
            printf("%8f ", xVector[x]);
        }
        printf("\n");
    }

    // -----------------------------------------------------[ reload Result ]--
    // Vector Result -- LOAD
    double result[dimOfMatrix];
    MPI_File_open(MPI_COMM_WORLD, pathToResultFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
    MPI_File_read(fhandle, &result, dimOfMatrix, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);

    if (my_rank == 0)
    {
        printf("Result (reloaded): \n");
        for (int s = 0; s < dimOfMatrix; s++)
        {
            printf("%f ", result[s]);
        }
        printf("\n");
    }

    HIT END*/
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
        //xprintf("mpicc -o ./app1 ./2my-mpi.c && mpiexec -f ./hosts -n 4 ./app1 -m Matrix_A_8x8 -v Vector_b_8x -e 0.0000000001\n");
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
