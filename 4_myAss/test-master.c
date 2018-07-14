/* manager */
#include <sys/select.h>

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

int my_rank, world_size; //MPI-STUFF
char *pathmaster_ori_matrixA;
char *pathmaster_ori_matrixB;
char *pathmaster_ori_matrixC;

int err, i;

void mutex()
{
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(400);
}

void h_printQuaMatrixOfDouble(char tag, double *matrix, int sizeOfMatrix, int rank, int rankToPrint)
{
    if (rank == rankToPrint)
    {
        printf("\n[%d]%c", rank, tag);
        for (int i = 0; i < (sizeOfMatrix * sizeOfMatrix); i++)
        {
            if (i % sizeOfMatrix == 0)
                printf("\n");
            printf("(%3d)%3.3f ", i, matrix[i]);
        }
        if (rank == 0)
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
        printf("*Parameter -a <path to picture>   :    Path to matrix a.\n");
        printf("*Parameter -b <path to picture>   :    Path to matrix b.  \n");
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
    int man_a = -1;
    int man_b = -1;
    int man_c = -1;

    opterr = 0;
    while ((c = getopt(argc, argv, "ha:b:c:")) != -1)
        switch (c)
        {
        case 'h':
            h_rootPrintHelp(my_rank);
            exit(0);
            break;
        case 'a':
            pathmaster_ori_matrixA = optarg;
            man_a = 0;
            break;
        case 'b':
            pathmaster_ori_matrixB = optarg;
            man_b = 0;
            break;
        case 'c':
            pathmaster_ori_matrixC = optarg;
            man_c = 0;
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
    int res = man_a + man_b + man_c;
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
    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
}

/**
 * @brief Calcs the result of multiplying two nxn-matrixes.
 * 
 * @param matrix_a 
 * @param matrix_b 
 * @param matrix_c The result is calculated in-place.
 * @param size 
 */
void seq_MatrixMulti(double *matrix_a, double *matrix_b, double *matrix_c, int size)
{
    int i, j, k;
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            for (k = 0; k < size; k++)
                matrix_c[i * size + j] += matrix_a[i * size + k] * matrix_b[k * size + j];
}
void error(char *mes) { printf("%s", mes); }

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // initializing of MPI-Interface
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    h_setAndCheckParams(argc, argv);
    MPI_File mpi_fileA;
    MPI_File mpi_fileB;
    MPI_Offset fsizeA;
    MPI_Offset fsizeB;
    int elemsToHandleA, elemsToHandleB;

    err = MPI_File_open(MPI_COMM_SELF, pathmaster_ori_matrixA, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileA);
    MPI_File_get_size(mpi_fileA, &fsizeA);
    err = MPI_File_open(MPI_COMM_SELF, pathmaster_ori_matrixB, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &mpi_fileB);
    MPI_File_get_size(mpi_fileB, &fsizeB);

    elemsToHandleA = fsizeA / (sizeof(double));
    elemsToHandleB = fsizeB / (sizeof(double));
    if (my_rank == 0)
        printf("Elms per row <%d>\n", elemsToHandleA);

    double *master_ori_matrixA = malloc(sizeof(double) * elemsToHandleA);
    double *master_ori_matrixB = malloc(sizeof(double) * elemsToHandleB);
    double *master_ori_matrixC = malloc(sizeof(double) * elemsToHandleB);

    int matrixDim = (int)sqrt(elemsToHandleA);

    MPI_File_read(mpi_fileA, master_ori_matrixA, elemsToHandleA, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(mpi_fileB, master_ori_matrixB, elemsToHandleB, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_fileA);
    MPI_File_close(&mpi_fileB);

    mutex();
    h_printQuaMatrixOfDouble('A', master_ori_matrixA, matrixDim, my_rank, 0);
    mutex();

    // Child section
    int sizeOfChilds = elemsToHandleA;
    char *program = "./test-worker-prog";
    MPI_Comm child;
    int spawnError[sizeOfChilds];
    MPI_Comm_spawn(program, MPI_ARGV_NULL, sizeOfChilds, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child, spawnError);
    int myid, flags = 0;
    MPI_Request request;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    double *local_matrixA[1], *local_matrixB[1];
    MPI_Ibcast(&matrixDim, 1, MPI_INT, MPI_ROOT, child, &request);

    MPI_Iscatter(master_ori_matrixA, 1, MPI_DOUBLE, local_matrixA, 1, MPI_DOUBLE, MPI_ROOT, child, &request);
    MPI_Iscatter(master_ori_matrixB, 1, MPI_DOUBLE, local_matrixB, 1, MPI_DOUBLE, MPI_ROOT, child, &request);

    flags = 0;
    while (flags == 0)
        MPI_Test(&request, &flags, &status);

    //FINAL Condition
    MPI_Request finalRequest;
    int finalStatus = -1;
    MPI_Ibcast(&finalStatus, 1, MPI_INT, MPI_ROOT, child, &finalRequest);
    flags = 0;
    while (flags == 0)
        MPI_Test(&finalRequest, &flags, &status);
    printf("Master rip\n");
    MPI_Finalize();
    return 0;
}