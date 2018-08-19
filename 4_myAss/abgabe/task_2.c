#include "mpi.h"    // import of the MPI definitions
#include "string.h" // import string for strcmp
#include <stdio.h>  // import of printf
#include <stdlib.h> // import stdlib for atoi
#include <math.h>   // import of sqrt

void start()
{
    char filename_A[256] = {0}, filename_B[256] = {0}, output_filename[256] = {0};
    FILE *f;

    //----------------//
    // Read filenames //
    //----------------//
    printf("\nPlease insert the file path to the first matrix A: ");
    scanf("%s", filename_A);
    printf("%s\n", filename_A);
    printf("\nPlease insert the file path to the second matrix B: ");
    scanf("%s", filename_B);
    printf("%s\n", filename_B);
    printf("\nPlease insert the file path for the output file: ");
    scanf("%s", output_filename);
    printf("%s\n", output_filename);

    //---------------//
    // Read matrix A //
    //---------------//
    f = fopen(filename_A, "rb"); // r for read, b for binary
    if (f == NULL)
    { // Check for success opening file B
        printf("Error: Coul not open file %s. Returning.\n", filename_A);
        return;
    }
    fseek(f, 0, SEEK_END); // Seek end of file
    int size = ftell(f);   // Get current file pointer position to calculate size
    fseek(f, 0, SEEK_SET); // Set pointer back to begin of file for reading
    int elements_A = size / sizeof(double);
    double matrix_h_A[elements_A];

    fread(matrix_h_A, sizeof(matrix_h_A), 1, f);
    fclose(f);

    //---------------//
    // Read matrix B //
    //---------------//
    f = fopen(filename_B, "rb"); // r for read, b for binary
    if (f == NULL)
    { // Check for success opening file B
        printf("Error: Coul not open file %s. Returning.\n", filename_A);
        return;
    }
    fseek(f, 0, SEEK_END); // See reading file A
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int elements_B = size / sizeof(double);
    double matrix_h_B[elements_B];

    fread(matrix_h_B, sizeof(matrix_h_B), 1, f);
    fclose(f);

    // Check that matrices have the same size
    int elements;
    if (elements_A == elements_B)
    {
        elements = elements_A;
    }
    else
    {
        printf("Error: Dimension %d of Matrix A does not match dimension %d of matrix B. Aborting!\n", elements_A, elements_B);
        return;
    }

    int num_procs = sqrt(elements); // For cannons algorithm the number of processes shall equal matrices rows and columns

    // Copy matrices from 1D array into a 2D array for Scattering with MPI_Type_create_subarray
    double matrix_B[num_procs][num_procs], matrix_A[num_procs][num_procs];
    for (int x = 0; x < num_procs; ++x)
    {
        for (int y = 0; y < num_procs; ++y)
        {
            matrix_B[x][y] = matrix_h_B[x * num_procs + y];
            matrix_A[x][y] = matrix_h_A[x * num_procs + y];
        }
    }

    //-----------------------//
    // Spawn child processes //
    //-----------------------//
    int errcodes[num_procs];
    MPI_Comm intercomm;

    char *command;
    command = "./cannon"; // Spawned processes will execute programm eith name cannon.

    int sub_matrix_size = num_procs / sqrt(num_procs);
    int sub_matrix_elements = sub_matrix_size * sub_matrix_size;

    // Spawn processes
    MPI_Comm_spawn(command, NULL, num_procs, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);

    // Broadcast necessary data to spawned processes
    MPI_Bcast(&sub_matrix_elements, 1, MPI_INT, MPI_ROOT, intercomm);
    MPI_Bcast(&output_filename, 256, MPI_CHAR, MPI_ROOT, intercomm);

    //--------------//
    // Scatter data //
    //--------------//
    // Create subarray type
    int complete_array_dims[2] = {num_procs, num_procs};
    int sub_array_dims[2] = {sub_matrix_size, sub_matrix_size};
    int start_array[2] = {0, 0};
    MPI_Datatype sub_array_type, sub_array_resized;
    MPI_Type_create_subarray(2, complete_array_dims, sub_array_dims, start_array, MPI_ORDER_FORTRAN, MPI_DOUBLE, &sub_array_type);
    MPI_Type_commit(&sub_array_type);

    MPI_Type_create_resized(sub_array_type, 0, sub_matrix_size * sizeof(double), &sub_array_resized);
    MPI_Type_commit(&sub_array_resized);

    // Create displacement array. Without these displacements Scatter works linewise and the subarrays overlap.
    double recv_buf[sub_matrix_elements];
    int displs[num_procs];
    int sends[num_procs];
    int displ = -1;
    for (int i = 0; i < sub_matrix_size; ++i)
    {
        for (int j = 0; j < sub_matrix_size; ++j)
        {
            displ += 1;
            displs[i * sub_matrix_size + j] = displ;
        }
        for (int j = 0; j < sub_matrix_size; ++j)
        {
            sends[i * sub_matrix_size + j] = 1;
        }
        displ += (sub_matrix_size - 1) * sub_matrix_size;
    }

    // Scatter matrix_A and matrix_B to child processes
    MPI_Scatterv(matrix_A, sends, displs, sub_array_resized, recv_buf, sub_matrix_elements, MPI_DOUBLE, MPI_ROOT, intercomm);
    MPI_Scatterv(matrix_B, sends, displs, sub_array_resized, recv_buf, sub_matrix_elements, MPI_DOUBLE, MPI_ROOT, intercomm);
}

int main(int argc, char *argv[])
{
    char user_input[32];
    for (int i = 0; i < argc; i++)
    {
        // print help if command line argument -h exists
        if (!strcmp(argv[i], "-h"))
        {
            printf("\nThis program calculates the product of two quadratic matrices with the Cannon Algorithm.\n\n"
                   "Execute the program with mpiexec.\n\n"
                   "Necessary parameters are:\n"
                   "-np defines the number of processes. Only 1 is valid here.\n\n"
                   "Optional additional parameters are:\n"
                   "-f specifies a hostfile. If no hostfile is given, the default hostfile is used.\n\n"
                   "An execution might look as follows:\n"
                   "mpiexec -np 1 -f hosts_ethernet ./task_2\n\n");
            return 0; // return after printing help. I assumed that someone who needs help does'nt wannt to execute the rest of the program in the same call.
        }
        if (!strcmp(argv[i], "-np"))
        {
            if (i + 1 < argc)
            {
                printf("Error: Parameter -np defined but no value given. Aborting.\n");
                return 1;
            }
            else
            {
                int np = atoi(argv[i + 1]);
                if (np > 1)
                {
                    printf("Error: Program can be executed with one process only. Aborting.\n");
                    return 1;
                }
            }
        }
    }

    fd_set rfds;
    int retval, len;
    char buff[255] = {0};

    /* Watch stdin (fd 0) to see when it has input. */
    FD_ZERO(&rfds);
    FD_SET(0, &rfds);
    MPI_Init(&argc, &argv);

    while (1)
    {
        printf("\nPlease type \"quit\" to close program or \"start\" to start a computation: ");
        select(1, &rfds, NULL, NULL, NULL);
        fgets(user_input, sizeof(user_input), stdin);
        int len = strlen(user_input) - 1;
        if (user_input[len] == '\n')
            user_input[len] = '\0';
        if (!strcmp(user_input, "quit"))
        {
            break;
        }
        else if (!strcmp(user_input, "start"))
        {
            start();
        }
        else
        {
            printf("Unknown command \"%s\".\n", user_input);
        }
    }
    MPI_Finalize(); //MPI_Finalize() should block until all processes - including the spawned child processes - have returned. In this case i tested it and it worked.
}
