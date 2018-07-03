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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char str[100];

    MPI_File mpi_fileA;
    MPI_File mpi_fileB;
    MPI_Offset fsizeA;
    MPI_Offset fsizeB;

    // -------------------------------------------------------[ CODE ]--

    //###############
    // root reads 2 paths  (A&B matrix)
    // root listen to start and quit.
    //###############

    //###############
    // Calculate number of nodes and starts the slaves
    //###############

    //###############
    // Init infrastructure and distribute the values
    // MPI_Cart_create()
    // Each process gets m^2 values where m = n/ srt(p)
    // MPI_Sendrecv_replace and MPI_Card_shift - HOW communication works.
    //###############

    //###############
    // All workers writing their values to file
    // Processing stops if masters get quit()
    // Listening if everything is done works by test!
    //###############

    // -------------------------------------------------------[ RESULT SAVE ]--
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