
#include <stdio.h>
#include "mpi.h"
#include <math.h>

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
    double bufferMatrix[elemsToHandle];
    MPI_File_read_ordered(fhandle, bufferMatrix, elemsToHandle, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);

    double bufferVector[dimOfMatrix];
    MPI_File_open(MPI_COMM_WORLD, pathToVector, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
    MPI_File_read(fhandle, bufferVector, dimOfMatrix, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fhandle);

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
    int isDDM = 1;
    //Checks if matrix is dominant.
    for (int xx = 0; xx < world_size; xx++)
    {

        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
        if (my_rank == xx)
        {
            for (int x = 0; x < blocksToHandle; x++)
            {
                printf("ROW [%d] ", x);
                for (int y = 0; y < dimOfMatrix; y++)
                {
                    printf("%f ", matrixs[x][y]);
                }
                printf("\n");
            }

            //check if dominant START
            int pushIndexByRank = my_rank * blocksToHandle;
            for (int i = 0; i < blocksToHandle; i++)
            {
                int pushIBy = i + pushIndexByRank;
                // for each column, finding sum of each row.
                int sum = 0;
                for (int j = 0; j < dimOfMatrix; j++)
                    sum += abs(matrixs[i][j]);

                // removing the diagonal element.
                sum -= abs(matrixs[i][pushIBy]);
                // checking if diagonal element is less
                // than sum of non-diagonal element.
                if (abs(matrixs[i][pushIBy]) < sum)
                    isDDM = 0;
                printf("[Row of interest %d: Value (%f)]\n", i, matrixs[i][pushIBy]);
            }
            if (isDDM == 0)
                printf("[Node %d]Matrix is not dominant.\n", my_rank);
            else
                printf("[Node %d]Matrix is dominant.\n", my_rank);
        }
        //check if dominant END
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    if (isDDM == 0)
    {
        printf("[Node %d] Part of matrix is not dominant. Calculation will be aborted.");
        exit(1);
    }
    //create dummy x vector *easy as hell

    //start calc.  *could be ripped

    //print output. *vector print should be easy

    MPI_Finalize(); // finalizing MPI interface
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
int isDDM(double m[i][j], int n)
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
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
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
        printf("%s\n", mes);
    }
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