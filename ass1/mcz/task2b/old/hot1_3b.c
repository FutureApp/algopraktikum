/***********************************************************************
 Program: hellomp.c                                                  
 Author: Michael Czaja, xxxxxxx                                           
 matriclenumber: 4293033                                             
 Assignment : 1                                                      
 Task: 2                                                             
 Parameters: no                                                      
 Environment variables: no                                           
                                                                     
 Description:                                                        
                                                                     
 Implements the odd-even tranposition sort - algorithm.               
 First, every computer will generate a random number. After that,
 the number of sequence will be sorted by the network due a simple parallel 
 strategy.

 Program uses buffered communication.
/************************************************************************/

#include "mpi.h"	// import of the MPI definitions
#include <stdio.h>  // import of the definitions of the C IO library
#include <string.h> // import of the definitions of the string operations
#include <unistd.h> // standard unix io library definitions and declarations
#include <errno.h>  // system error numbers

#include <stdlib.h>

#define MAX_BUFFER_SIZE 1000

int checkParameters(int my_rank, int argc, char *argv[]);
int terminateIfNeeded(int terminate);
int rootGetAndPrintNodeRandInRingMode(char tag[], int commLine, int myRank, int myRand);
int terminateIfNeededSilently(int terminate);
int printHelp();
int compare(const void *a, const void *b);

void otPrint(int rankWhichPrints, int my_rank, char message[]);
void printArray(int array[], int size);
void otPrintWRank(int rankWhichPrints, int my_rank, char message[]);

int world_size;
int mValue;
int commLine = 9999;

int main(int argc, char *argv[])
{
	// -----------------------------------------------------------------[Init]--
	int namelen;							 // length of name
	int my_rank;							 // rank of the process
	MPI_Init(&argc, &argv);					 // initializing of MPI-Interface
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //get your rank

	// init call
	srand(time(NULL)); // should only be called once
	int mesCommline = 99999;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	char buffer[(MAX_BUFFER_SIZE * world_size) + MPI_BSEND_OVERHEAD];
	int bsize = sizeof(buffer);

	// ----------------------------------------------------------[ Para check ]--

	int needToTerminate = checkParameters(my_rank, argc, argv);

	// ---------------------------------------------------------[ Data Create ]--

	int myNumbers[mValue];
	int neighNumbers[mValue];
	int meAndNeigh[2 * mValue];
	for (int i = 0; i < mValue; i++)
	{
		int myRand = rand() % (100 + 1 - 1) + 1;

		char szBuffer[4];
		sprintf(szBuffer, "%d", myRand);
		myNumbers[i] = myRand;
	}

	qsort(myNumbers, mValue, sizeof(int), compare);

	// ----------------------------------------------------------------[ Main ]--

	char *c,
		proc_name[MPI_MAX_PROCESSOR_NAME + 1]; // hostname

	memset(proc_name, 0, MPI_MAX_PROCESSOR_NAME + 1);
	// initialize string with NULL-characters
	MPI_Get_processor_name(proc_name, &namelen);
	// finding out own computer name

	if ((c = strchr(proc_name, '.')) != NULL)
		*c = '\0';

	// ------------------------------------------------------------[Init Call]--

	// ------------------------------------------------------------[Para Part]--
	MPI_Buffer_attach((void *)buffer, bsize * world_size);
	for (int round = 0; round < world_size; round++)
	{
		if (round % 2 == 0)
		{
			if (my_rank % 2 == 0)
			{
				if (my_rank + 1 < world_size)
				{
					MPI_Sendrecv(&myNumbers, mValue, MPI_INT, my_rank + 1, round, &neighNumbers, mValue, MPI_INT, my_rank + 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else
				{
					// Do nothing no right neighbour.
				}
			}
		}
	}

	// ----------------------------------------------------------[Result Call]--

	// Print everything in right order (0 -> n) -- RING-Call
	//printResultBySendingThemToRoot(my_rank, myNumbers);

	//-----------------------------------------------------------------[ END ]--
	MPI_Buffer_detach((void *)buffer, &bsize);
	MPI_Finalize(); // finalizing MPI interface

	return 0; // end of progam with exit code 0
}

// ------------------------------------------------------------------[ UTILS ]--

void printResultBySendingThemToRoot(int rank, int array[])
{
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(100);
	if (rank == 0)
	{
		printf("Result: ");
	}
	for (int i = 0; i < world_size; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == i)
			printArray(array, mValue);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(100);
	printf("\n");
}

void printArray(int array[], int size)
{
	int index;
	for (index = 0; index < size; index++)
	{
		printf("%d ", array[index]);
	}
}

/**
	 * @brief  
	 * Checks if all parameters are given and sets them globally for the flowing execution. 
	 * If -h tag is detacted then a help-message will be printend and the execution will be stopped.
	 * 
	 * Returns 1 if -h is found.
	 * 
	 * @param my_rank Rank of the processor-
	 * @param argc Number of args.
	 * @param argv Args.
	 * @return int  1 if  termination is needed.
	 */
int checkParameters(int my_rank, int argc, char *argv[])
{
	int terminate = 0;
	int parameterMFound = 0;
	for (int i = 0; i < argc; i++)
	{
		if ((strcmp(argv[i], "-m") == 0))
		{
			if (argv[i + 1])
			{
				mValue = atoi(argv[i + 1]);
				parameterMFound = 1;
			}
			else
			{
				if (my_rank == 0)
				{
					printf("Problem occurs while trying to cast input m <%s> to int", argv[i + 1]);
					printHelp();
				}
				terminate = 1;
			}
		}
		if (strcmp(argv[i], "-h") == 0)
		{
			if (my_rank == 0)
			{
				printHelp();
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
			printf("Problem occurs couldn't find parameter m. Checkout help-section.");
			printHelp();
		}
		terminate = 1;
	}
	if (terminate == 1)
	{
		exit(0);
	}

	if (mValue < 1)
	{
		otPrint(0, my_rank, "God bless, nothing to do. Thanks to you!\n");
	}
	return terminate;
}

/**
 * @brief Prints a messag to the screen only onces.  * 
 * @param rankWhichPrints  Which rank should print the message.
 * @param my_rank Rank of the machine.
 * @param message Message to print.
 */
void otPrint(int rankWhichPrints, int my_rank, char message[])
{
	if (my_rank == rankWhichPrints)
	{
		printf("%s ", message);
	}
}

void otPrintWRank(int rankWhichPrints, int my_rank, char message[])
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
int terminateIfNeeded(int terminate)
{
	if (terminate == 1)
	{
		printf("Execution will be canceled");
		exit(0);
	}
}

/**
 * @brief 
 * Canceled the program-execution if needed.
 * 
 * @param terminate 1- to stop execution.
 */
int terminateIfNeededSilently(int terminate)
{
	if (terminate == 1)
	{
		exit(0);
	}
}
int printHelp()
{
	printf("\n");
	printf("-----------------------------------------------------[Help]--\n");
	printf("\n");
	printf("Program is optimized for less then  99998 given nodes. \n");
	printf("Every parameter marked with <*> is required!\n");

	printf("Parameter* -m <number>: Specifies the amount of randome numbers each process need to generate.\n");
	printf("\n");
	printf("\n");
}

/**
 * @brief 
 * 
 * Compares to diffent integers.
 * Source https: //stackoverflow.com/questions/3893937/c-array-sorting-tips.
 * @param a 
 * @param b 
 * @return int  0 - if same ; -1 if a < b; 1 if a>b;
 */
int compare(const void *a, const void *b)
{
	int int_a = *((int *)a);
	int int_b = *((int *)b);

	if (int_a == int_b)
		return 0;
	else if (int_a < int_b)
		return -1;
	else
		return 1;
}
