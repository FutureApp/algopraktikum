/***********************************************************************
 Program: my-mpi-jacobi.c
 Author: Michael Czaja, Muttaki Aslanparcasi
 matriclenumber: 4293033, 5318807
 Assignment : 3
 Task: 2

 Description:
MPI program that solves a set of linear equations Ax = b with the Jacobi method that
converges if the distance between the vectors x^(k) and x^(k+1) is small enough. (Calc by Cols)
The special feature of this programm is that the Matrix is scattered in parts as a block of columns.
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

char *pathToMatrix; // IN - -pm
char *pathToVector; // IN - -pv
double eps;         // IN - -eps

int my_rank, world_size; //MPI-STUFF

double distanceV(double xOld[], double xNew[], int numberOfCols);
double calcDif(double xOld[], double xNew[], int numberOfCols);
void *printVector(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint);
void *printVectorNoBar(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint);

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
    //check if dominant START
    int isLocalPartDDM = 1;
    for (int i = 0; i < blocksToHandle; i++)
    {
        // for each column, finding sum of each row.
        int sum = 0;
        for (int j = 0; j < dimOfMatrix; j++)
            sum += abs(bufferMatrix[i + j * dimOfMatrix]);

        // removing the diagonal element.
        sum -= abs(bufferMatrix[i + i * dimOfMatrix]);
        // checking if diagonal element is less
        // than sum of non-diagonal element.
        if (abs(bufferMatrix[i + i * dimOfMatrix]) < sum)
            isLocalPartDDM = 0;
    }
    if (isLocalPartDDM == 0)
    {
        printf("[Node %d]Matrix is not dominant. Exec. stops now.\n", my_rank);
        abort();
    }
    else
        printf("[Node %d]Matrix is diagonal dominant.\n", my_rank);

    // --------------------------------------------------[ Distribute INPUT ]--
    MPI_Datatype vector1, vector2;

    int linesOfMatrix = dimOfMatrix;
    int columnsOfMatrixForProc = dimOfMatrix / world_size;
    int elemsToHandleEach = columnsOfMatrixForProc * columnsOfMatrixForProc; //handles elements of quadratic Matrix

    MPI_Type_vector(linesOfMatrix, columnsOfMatrixForProc, linesOfMatrix, MPI_DOUBLE, &vector1);
    MPI_Type_commit(&vector1);
    MPI_Type_create_resized(vector1, 0, columnsOfMatrixForProc * sizeof(double), &vector2);
    MPI_Type_commit(&vector2);

    double *new_matrixBuffer = malloc(sizeof(double) * dimOfMatrix * dimOfMatrix);
    double *new_matrixA = malloc(sizeof(double) * dimOfMatrix * dimOfMatrix);

    new_matrixA = bufferMatrix;
    MPI_Scatter(new_matrixA, 1, vector2, new_matrixBuffer, linesOfMatrix * columnsOfMatrixForProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //distributes the Matrix to the corresponding nodes

    double *new_vectorBuffer = malloc(sizeof(double) * dimOfMatrix);
    double *new_vectorB = malloc(sizeof(double) * dimOfMatrix);

    int numOfElemVec = dimOfMatrix / world_size;
    new_vectorB = bufferVector;
    MPI_Scatter(new_vectorB, numOfElemVec, MPI_DOUBLE, new_vectorBuffer, numOfElemVec, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //distributes the Vector to the corresponding nodes

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

    // DEBUGING purpose  START
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
    } // DEBUGING purpose END

    // Creates the x-vecs.
    double *new_vectorX = malloc(sizeof(double) * dimOfMatrix);
    double *old_vectorX = malloc(sizeof(double) * dimOfMatrix);
    for (i = 0; i < dimOfMatrix; i++)
    {
        new_vectorX[i] = 0;
        old_vectorX[i] = 1;
    }

    // ------------------------------------------------------------[ Jacobi ]--

    int calcIsFinished = 0;
    int iterationCount = 0;
    double lastEps = -1;

    while (calcIsFinished == 0)
    {
        if (my_rank == 0)
            printf("[%d] ---------------------------- [iteration %d] lastEps(%f)\n", my_rank, iterationCount, lastEps);
        printVector('I', old_vectorX, dimOfMatrix, my_rank, 0);

        int nodeHoldsA = 0;
        int innerCounter = 0;

        // for each row in the matrix.
        for (i = 0; i < dimOfMatrix; i++)
        {
            double localSum = 0;
            // calcs for each col the local sum.
            for (x = 0; x < blocksToHandle; x++)
            {
                double matrixValue = new_matrixBuffer[x + (blocksToHandle * i)];
                double xValue = old_vectorX[x + my_rank * blocksToHandle];
                localSum += matrixValue * xValue;
            }

            // dets. node witch the dia. elem.
            if ((i % blocksToHandle) == 0 && (i != 0))
                nodeHoldsA++;

            // ------------------------------------------------------------[ Reduce ]--
            double world_sum = 0;
            MPI_Reduce(&localSum, &world_sum, 1, MPI_DOUBLE, MPI_SUM, nodeHoldsA, MPI_COMM_WORLD);

            // ----------------------------------------------------------[ Node DIA ]--
            // calcs the new x-vec entry. Calc done by node with dia. elem.
            double diaElem = 0;
            double Xki = 0;
            if (my_rank == nodeHoldsA)
            {
                int diaElemPos = (i % blocksToHandle) + (i * blocksToHandle);
                diaElem = new_matrixBuffer[diaElemPos];
                world_sum -= diaElem * old_vectorX[i]; // removes dia-elem. Prev added in reduce-step.

                int posInB = diaElemPos % blocksToHandle;
                double valueOfB = new_vectorBuffer[posInB];
                // new vec - x - entry
                Xki = (1 / diaElem) * (valueOfB - world_sum);
            }

            MPI_Bcast(&Xki, 1, MPI_DOUBLE, nodeHoldsA, MPI_COMM_WORLD);
            new_vectorX[i] = Xki;
        } // calc newX-vector END

        // ----------------------------------------------------------[ Checking ]--
        innerCounter = 0;
        nodeHoldsA = 0;
        double curEps = calcDif(old_vectorX, new_vectorX, dimOfMatrix);

        // copy vals and cleans new vec - x.
        for (int gg = 0; gg < dimOfMatrix; gg++)
        {
            old_vectorX[gg] = new_vectorX[gg];
            new_vectorX[gg] = 0;
        }

        if (curEps < eps)
            calcIsFinished = 1;

        lastEps = curEps;
        iterationCount++;
    }

    // ------------------------------------------------------------[ RESULT ]--

    double *sendPartX = malloc(sizeof(double) * blocksToHandle);
    for (int no = 0; no < blocksToHandle; no++)
    {
        int pos = no + my_rank * blocksToHandle;
        sendPartX[no] = old_vectorX[pos];
        printf("RES [%d : %d] %f %f\n", my_rank, pos, sendPartX[no], old_vectorX[pos]);
    }

    double *revPartX = malloc(sizeof(double) * blocksToHandle * world_size);

    MPI_Gather(sendPartX, blocksToHandle, MPI_DOUBLE, revPartX, blocksToHandle, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        printf("------------------------------------------[ Result ]\n");
    printVector('L', revPartX, dimOfMatrix, my_rank, 0);

    // -------------------------------------------------------[ RESULT SAVE ]--
    // write and reload result.
    if (my_rank == 0)
    {
        char *pathToResultFile = "./rest2"; //PATH where to save result
        err = MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fhandle);
        if (err)
            printf("Error opening the file. \n");
        MPI_File_write(fhandle, revPartX, (blocksToHandle * world_size), MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);
        printf("Result saved. Check < %s >.\n", pathToResultFile);

        double reloadX[dimOfMatrix];
        MPI_File_open(MPI_COMM_SELF, pathToResultFile, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fhandle);
        MPI_File_read(fhandle, &reloadX, dimOfMatrix, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&fhandle);
        printf("------------------------------------------[Result rel.]\n");
        printVectorNoBar('L', reloadX, dimOfMatrix, my_rank, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1000);
    printf("[node %d] ExEnd.\n", my_rank);
    MPI_Finalize(); // finalizing MPI interface
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
void *printVectorNoBar(char tag, double *vector, int dimOfVec, int yourRank, int rankToPrint)
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
        printf("------------------------------------[ HELP ]\n");
        printf("*Parameter -m <path to matrix>   :    Path to file containing the matrix-entrys.\n");
        printf("*Parameter -v <path to vector>   :    Path to file containing the vector-entrys.\n");
        printf("*Parameter -e <number as double>:     Specifies epsilon. Need to be double. \n");
        printf("\n");
        printf("Example call:\n");
        printf("mpicc -o ./app1 ./2my-mpi.c && mpiexec -f ./hosts -n 4 ./app1 -m Matrix_A_8x8 -v Vector_b_8x -e 0.0000000001\n");
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
    int man_m = -1;
    int man_v = -1;
    int man_e = -1;

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
            man_m = 0;
            break;
        case 'v':
            pathToVector = optarg;
            man_v = 0;
            break;
        case 'e':
            sscanf(optarg, "%lf", &eps);
            man_e = 0;
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
    int res = man_e + man_m + man_v;
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
    printf("INPUT(%dr): m = %s, v = %s, eps = %f\n",
           my_rank, pathToMatrix, pathToVector, eps);

    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
}

