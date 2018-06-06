/***********************************************************************
 Program: hellomp.c                                                  
 Author: Michael Czaja, Muttaki Aslanparcasi                                           
 matriclenumber: 4293033, 5318807                                             
 Assignment : 1                                                      
 Task: 3                                                             
 Parameters: no                                                      
 Environment variables: no                                           
                                                                     
 Description:                                                        
In this Assignment there is a method for numerical Integration.
Here we use the Romberg Method combinded with MPI parallel programming.


/************************************************************************/

#include "mpi.h"	// import of the MPI definitions
#include <stdio.h>  // import of the definitions of the C IO library
#include <string.h> // import of the definitions of the string operations
#include <unistd.h> // standard unix io library definitions and declarations
#include <errno.h>  // system error numbers
#include <math.h>
#include <stdlib.h>

#define MAX_BUFFER_SIZE 1000

// UTIL
int utilCheckParameters(int my_rank, int argc, char *argv[]);
int utilTerminateIfNeeded(int terminate);
int utilTerminateIfNeededSilently(int terminate);
int utilPrintHelp();
void utilOTPrint(int rankWhichPrints, int my_rank, char message[]);
void utilPrintArray(double array[], int size);
void utilOTPrintWRank(int rankWhichPrints, int my_rank, char message[]);

int world_size;
int mValue;
int commLine = 9999;
int my_rank; // rank of the process

int main(int argc, char *argv[])
{
	// -----------------------------------------------------------------[Init]--
	double idleOperations = 1;
	double idleTime = 0;
	int steps = -1;
	int namelen; // length of name
	int my_rank; // rank of the process

	MPI_Init(&argc, &argv);					 // initializing of MPI-Interface
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	double mpi_programStart = MPI_Wtime();
	char buffer[(MAX_BUFFER_SIZE * world_size) + MPI_BSEND_OVERHEAD];
	int bsize = sizeof(buffer);

	// -----------------------------------------------------------[ pre Init ]--

	char *c, proc_name[MPI_MAX_PROCESSOR_NAME + 1]; // hostname
	memset(proc_name, 0, MPI_MAX_PROCESSOR_NAME + 1);
	// initialize string with NULL-characters
	MPI_Get_processor_name(proc_name, &namelen);
	// finding out own computer name
	if ((c = strchr(proc_name, '.')) != NULL)
		*c = '\0';

	// ------------------------------------------------------------[Init Call]--

	int n = -100;
	int i;
	double **R;
	double globalBoundA;
	double globalBoundB;

	// --------------------------------------------------------------[ Input ]--

	// ---------------------------------------------------------[ Para check ]--

	double mpi_loopStart = MPI_Wtime();
	// ------------------------------------------------------------[Para Part]--

	double mpi_loopEnd = MPI_Wtime();
	double loopTimaAtAll = mpi_loopEnd - mpi_loopStart;
	// --------------------------------------------------------[Para Part END]--
	double mpi_programEnd = MPI_Wtime();

	// ----------------------------------------------------------[Result Call]--
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(100);
	if (my_rank == 0)
	{
		utilOTPrint(0, my_rank, "--------------------- RESULT\n");
		utilOTPrint(0, my_rank, "--------------------- MES\n");
	}
	MPI_Buffer_detach((void *)buffer, &bsize);
	MPI_Finalize(); // finalizing MPI interface
	return 0;
}

// --------------------------------------------------------------[ TaskFunc. ]--

// ------------------------------------------------------------------[ UTILS ]--

/**
 * @brief Prints and array.
 * 
 * @param array The array.
 * @param size Size of array.
 */
void utilPrintArray(double array[], int size)
{
	int index;
	for (index = 0; index < size; index++)
	{
		printf("<%lf>\n", array[index]);
	}
}

/**
 * @brief
 * Checks if all parameters are given and sets them globally for the following execution.
 * If -h tag is detacted then a help-message will be printend and the execution will be stopped.
 *
 * Returns 1 if -h is found.
 *
 * @param my_rank Rank of the processor-
 * @param argc Number of args.
 * @param argv Args.
 * @return int  1 if  termination is needed.
 */
int utilCheckParameters(int my_rank, int argc, char *argv[])
{
	//TODO Parameter überprüfen. Checke hier die eingabe von den globalen Schranken a und b.
	// Orientiere dich hier wie an den parameter -m.
	int terminate = 0;
	int parameterMFound = 0;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			if (my_rank == 0)
			{
				utilPrintHelp();
				terminate = 1;
				break;
			}
			else
			{
				terminate = 1;
			}
		}
	}
	if (parameterMFound != 1)
	{
		if (my_rank == 0)
		{
			printf(
				"Problem occurs couldn't find parameter m. Checkout help-section.");
			utilPrintHelp();
		}
		terminate = 1;
	}

	if (mValue < 1)
	{
		terminate = 1;
		utilOTPrint(0, my_rank, "God bless, nothing to do. Thanks to you!\n");
	}
	if (taskIspowerof2(world_size) != 1)
	{
		terminate = 1;
		utilOTPrint(0, my_rank, "ERROR -----------------\n");
		utilOTPrint(0, my_rank, "The number of nodes isn't a power of 2. Please check the -n parameter.\n");
		utilPrintHelp();
	}

	if (terminate == 1)
	{
		exit(0);
	}
	return terminate;
}

/**
 * @brief Prints a messag to the screen only onces.  * 
 * @param rankWhichPrints  Which rank should print the message.
 * @param my_rank Rank of the machine.
 * @param message Message to print.
 */
void utilOTPrint(int rankWhichPrints, int my_rank, char message[])
{
	if (my_rank == rankWhichPrints)
	{
		printf("%s", message);
	}
}

/**
 * @brief Prints message. Message contains the node-rank.
 * 
 * @param rankWhichPrints Node, which should print.
 * @param my_rank rank of the node.
 * @param message  message to print.
 */
void utilOTPrintWRank(int rankWhichPrints, int my_rank, char message[])
{
	if (my_rank == rankWhichPrints)
	{
		printf("%d : %s ", my_rank, message);
	}
}

/**
 * @brief 
 * Cancels the program-execution if needed.
 * 
 * @param terminate If 1 then stop execution.
 */
int utilTerminateIfNeeded(int terminate)
{
	if (terminate == 1)
	{
		printf("Execution will be canceled");
		exit(0);
	}
}

/**
 * @brief 
 * Aborts the program-execution if needed.
 * 
 * @param terminate 1- to stop execution.
 */
int utilTerminateIfNeededSilently(int terminate)
{
	if (terminate == 1)
	{
		exit(0);
	}
}
/**
 * @brief Prints the help-message.
 * 
 * @return int 0 if successful.
 */
int utilPrintHelp()
{

	// TODO Passe die Help-Message an
	utilOTPrint(0, my_rank, "\n");
	utilOTPrint(0, my_rank, "-----------------------------------------------------[Help]--\n");
	utilOTPrint(0, my_rank, "\n");
	utilOTPrint(0, my_rank, "Program is optimized for less then  99998 given nodes. \n");
	utilOTPrint(0, my_rank, "MPI-Parameter -n must be a power of 2. \n");
	utilOTPrint(0, my_rank, "\n");
	utilOTPrint(0, my_rank, "Follow the instructions on the screen. Enter the first 2 arguments as double's (integration limits).\n");
	utilOTPrint(0, my_rank, "Last parameter(integer) should be between 1 and 15.\n");
	utilOTPrint(0, my_rank, "No guarantee if you differ the format! \n");
	utilOTPrint(0, my_rank, " \n");
	utilOTPrint(0, my_rank, "\n");
	utilOTPrint(0, my_rank, "\n");
	return 0;
}
