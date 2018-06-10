
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

    //create dummy x vector

    //check if dominant

    //start calc.

    //print output.

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
                    printf("%f ", buffer[x]);
                }
                else
                {
                    printf("%f ", buffer[x]);
                }
            }
            printf("|\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(200);
    }
    */

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