#include <sys/select.h>

#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, cart_comm_rank;
    MPI_Comm parent;
    MPI_Comm cart_comm;
    int coordinates[2];
    int left_neighbor, right_neighbor, top_neighbor, bottom_neighbor;
    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File file_handler;
    MPI_Status status;

    // Receive number of matrix elements and output filename
    int sub_matrix_elements;
    MPI_Bcast(&sub_matrix_elements, 1, MPI_INT, 0, parent);
    char output_filename[256];
    MPI_Bcast(&output_filename, 256, MPI_CHAR, 0, parent);

    int sub_matrix_size = sqrt(sub_matrix_elements);
    double matrix_A[sub_matrix_elements], matrix_B[sub_matrix_elements];

    // Receive matrix data
    double *result_matrix = (double *)calloc(sub_matrix_elements, sizeof(double));
    double dummy_matrix[sub_matrix_elements];
    MPI_Scatterv(dummy_matrix, sub_matrix_elements, 1, MPI_DOUBLE, matrix_A, sub_matrix_elements, MPI_DOUBLE, 0, parent);
    MPI_Scatterv(dummy_matrix, sub_matrix_elements, 1, MPI_DOUBLE, matrix_B, sub_matrix_elements, MPI_DOUBLE, 0, parent);

    int printer = 0, i = 0;
    if (rank == printer)
    {
        printf("My rank == %d\n", rank);
        for (i = 0; i < sub_matrix_elements; i++)
        {
            if (i % 2 == 0)
            {
                printf("\n");
            }
            printf("%.3f ", matrix_A[i]);
        }
    }

    // Get number of spawned child processes
    int num_procs;
    MPI_Comm_size(parent, &num_procs);

    // Create 2D cartesian communicator
    int dims[2] = {sub_matrix_size, sub_matrix_size};
    int wrap[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2 /*dimension*/, dims /*dim array*/, wrap /*wrap array*/, 1 /*reorder*/, &cart_comm);

    // Get rank and coordinate in new cartesian communicator
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coordinates);

    // Swap data if necessary
    MPI_Cart_shift(cart_comm, 1, coordinates[0], &left_neighbor, &right_neighbor);
    MPI_Cart_shift(cart_comm, 0, coordinates[1], &top_neighbor, &bottom_neighbor);
    MPI_Sendrecv_replace(&matrix_A, sub_matrix_elements, MPI_DOUBLE, left_neighbor, 1, right_neighbor, 1, cart_comm, &status);
    MPI_Sendrecv_replace(&matrix_B, sub_matrix_elements, MPI_DOUBLE, top_neighbor, 1, bottom_neighbor, 1, cart_comm, &status);

    // Calculate product on initial submatrix
    for (int element = 0; element < sub_matrix_elements; ++element)
    {
        int x = element % sub_matrix_size;
        int y = element / sub_matrix_size;
        for (int j = 0; j < sub_matrix_size; ++j)
        {
            result_matrix[element] += matrix_A[y * sub_matrix_size + j] * matrix_B[j * sub_matrix_size + x];
        }
    }

    for (int i = 0; i < sub_matrix_size - 1; i++)
    {
        // Shift data ..
        MPI_Cart_shift(cart_comm, 1, -1, &right_neighbor, &left_neighbor);
        MPI_Cart_shift(cart_comm, 0, -1, &bottom_neighbor, &top_neighbor);
        MPI_Sendrecv_replace(&matrix_A, sub_matrix_elements, MPI_DOUBLE, left_neighbor, i, right_neighbor, i, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&matrix_B, sub_matrix_elements, MPI_DOUBLE, top_neighbor, i, bottom_neighbor, i, cart_comm, MPI_STATUS_IGNORE);
        // .. and calculate product with the next submatrix
        for (int element = 0; element < sub_matrix_elements; ++element)
        {
            int x = element % sub_matrix_size;
            int y = element / sub_matrix_size;
            for (int j = 0; j < sub_matrix_size; ++j)
            {
                result_matrix[element] += matrix_A[y * sub_matrix_size + j] * matrix_B[j * sub_matrix_size + x];
            }
        }
    }

    //////////////////////////
    //    Write solution    //
    //////////////////////////
    // Create subarray type to write back subresults in global matrix
    int complete_array_dims[2] = {num_procs, num_procs};
    int sub_array_dims[2] = {sub_matrix_size, sub_matrix_size};
    int start_array[2] = {(coordinates[0] * sub_matrix_size), (coordinates[1] * sub_matrix_size)}; // The position in the global matrix corresponds to the coordinate in the communicator
    MPI_Datatype sub_array_type, sub_array_resized;
    MPI_Type_create_subarray(2, complete_array_dims, sub_array_dims, start_array, MPI_ORDER_C, MPI_DOUBLE, &sub_array_type);
    MPI_Type_commit(&sub_array_type);

    // Open file
    int error = MPI_File_open(cart_comm, output_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file_handler);
    if (error)
    {
        if (rank == 0)
        {
            char msg[MPI_MAX_ERROR_STRING];
            int resultlen;
            MPI_Error_string(error, msg, &resultlen);

            printf("Could not open File %s because of error \"%s\". Exiting.\n", output_filename, msg);
        }
        MPI_Finalize();
        return error;
    }

    // Set subarray as view
    MPI_File_set_view(file_handler, 0, MPI_DOUBLE, sub_array_type, "native", MPI_INFO_NULL);

    // Write file
    MPI_File_write_all(file_handler, result_matrix, sub_matrix_elements, MPI_DOUBLE, MPI_STATUS_IGNORE);

    // Close file
    MPI_File_close(&file_handler);

    /////////////////////////
    //    Check results    //
    /////////////////////////
    // Read binary file and convert it into txt file to check the results
    if (rank == 0)
    {
        double buffer[sub_matrix_elements * sub_matrix_elements];
        FILE *ptr;

        ptr = fopen(output_filename, "rb"); // r for read, b for binary
        fread(buffer, sizeof(buffer), 1, ptr);

        ptr = fopen(strcat(output_filename, ".txt"), "w");
        for (int i = 0; i < sub_matrix_elements; i++)
        {
            for (int j = 0; j < sub_matrix_elements; ++j)
            {
                fprintf(ptr, "%f", buffer[i * sub_matrix_elements + j]);
                if (j != sub_matrix_elements - 1)
                    fprintf(ptr, " ");
            }
            fprintf(ptr, "\n");
        }
        fclose(ptr);
    }

    MPI_Finalize();
    return 0;
}
