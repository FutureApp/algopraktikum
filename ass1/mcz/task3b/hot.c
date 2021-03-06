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

 Programm is using buffered communication.
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
int world_size;
int terminateIfNeededSilently(int terminate);

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

	// -----------------------------------------------------------[Para check]--

	int needToTerminate = checkParameters(my_rank, argc, argv);
	terminateIfNeededSilently(needToTerminate);

	// -----------------------------------------------------------------[Main]--

	char *c, proc_name[MPI_MAX_PROCESSOR_NAME + 1]; // hostname

	memset(proc_name, 0, MPI_MAX_PROCESSOR_NAME + 1);
	// initialize string with NULL-characters
	MPI_Get_processor_name(proc_name, &namelen);
	// finding out own computer name

	if ((c = strchr(proc_name, '.')) != NULL)
		*c = '\0';

	//rand() % (max_number + 1 - minimum_number) + minimum_number
	int myRand = rand() % (100 + 1 - 1) + 1;

	// ------------------------------------------------------------[Init Call]--
	rootGetAndPrintNodeRandInRingMode("Init", mesCommline, my_rank, myRand);

	// ------------------------------------------------------------[Para Part]--
	MPI_Buffer_attach((void *)buffer, bsize * world_size);
	for (int round = 0; round < world_size; round++)
	{
		if (round == 0)
		{
		}
		int numberFormOtherP = -1;
		int numberToSendBack = -1;

		//even-phase
		if (round % 2 == 0)
		{
			if (my_rank % 2 == 0)
			{
				if (my_rank + 1 >= world_size)
				{
					//Do Nothing no right neighbour
				}
				else
				{
					MPI_Recv(&numberFormOtherP, 1, MPI_INT, my_rank + 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (myRand >= numberFormOtherP)
					{
						numberToSendBack = myRand;
						myRand = numberFormOtherP;
					}
					else
					{
						numberToSendBack = numberFormOtherP;
					}
					MPI_Bsend(&numberToSendBack, 1, MPI_INT, my_rank + 1, round, MPI_COMM_WORLD);
					//	printf("Round <%d> | From <%d> to <%d> | data <%d>\n", round, my_rank, my_rank + 1, numberFormOtherP);
				}
			}
			if (my_rank % 2 == 1)
			{
				MPI_Bsend(&myRand, 1, MPI_INT, my_rank - 1, round, MPI_COMM_WORLD);
				MPI_Recv(&numberFormOtherP, 1, MPI_INT, my_rank - 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				myRand = numberFormOtherP;
				//	printf("Round <%d> | <%d> to <%d> | data <%d>\n", round, my_rank, my_rank - 1, round);
			}
		}
		//odd-phase
		else
		{
			if (my_rank % 2 == 1)
			{
				if (my_rank + 1 >= world_size)
				{
					//Do nothing because you don't have a right neighbour.
				}
				else
				{
					MPI_Recv(&numberFormOtherP, 1, MPI_INT, my_rank + 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (myRand >= numberFormOtherP)
					{
						numberToSendBack = myRand;
						myRand = numberFormOtherP;
					}
					else
					{
						numberToSendBack = numberFormOtherP;
					}
					MPI_Bsend(&numberToSendBack, 1, MPI_INT, my_rank + 1, round, MPI_COMM_WORLD);
				}
			}

			if (my_rank % 2 == 0)
			{
				if (my_rank == 0)
				{
					//Do Nothing
				}
				else
				{
					MPI_Bsend(&myRand, 1, MPI_INT, my_rank - 1, round, MPI_COMM_WORLD);
					MPI_Recv(&numberFormOtherP, 1, MPI_INT, my_rank - 1, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					myRand = numberFormOtherP;
				}
			}
		}
	}

	// ----------------------------------------------------------[Result Call]--

	// Print everything in right order (0 -> n) -- RING-Call
	rootGetAndPrintNodeRandInRingMode("Res.", mesCommline + 1, my_rank, myRand);

	//-----------------------------------------------------------------[ END ]--
	MPI_Buffer_detach((void *)buffer, &bsize);
	MPI_Finalize(); // finalizing MPI interface
	return 0;		// end of progam with exit code 0
}

/**
 * @brief 
 * Collects informations and prints them to the consol. Communications behaves
 * like ring-call. 
 * 
 * @param tag Tag-message
 * @param myRank Rank of node
 * @param myRand number or int var to print.
 * @return int 0: if successfull.
 */
int rootGetAndPrintNodeRandInRingMode(char tag[], int commLine, int myRank, int myRand)
{
	int dummy = 0;
	int firstTimeToPrint = 0;
	for (int i = 0; i < world_size; i++)
	{
		if (myRank == 0)
		{
			if ((firstTimeToPrint == 0) && i == 0)
			{
				printf("--------------------------[ %s ]--\n", tag);
				printf("Node|Number - %d|%d\n", myRank, myRand);
				firstTimeToPrint++;
			}
			else
			{
				MPI_Recv(&dummy, 1, MPI_INT, i, commLine, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("Node|Number - %d|%d\n", i, dummy);
			}
		}
		else
		{
			MPI_Send(&myRand, 1, MPI_INT, 0, commLine, MPI_COMM_WORLD);
		}
	}
	if (myRank == 0)
		printf("-----------------------------------|\n");

	return 0;
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
	 * @return int  1 - if -h tag or mismatch of parameters. 
	 * 				0 - otherwise.
	 */
int checkParameters(int my_rank, int argc, char *argv[])
{
	int terminate = 0;
	for (int i = 0; i < argc; i++)
	{
		// printf("Arguments %d : %s\n", i, argv[i]);
		if (strcmp(argv[i], "-h") == 0)
		{
			if (my_rank == 0)
			{
				printf("\n");
				printf("-----------------------------------------------------[Help]--\n");
				printf("\n");
				printf("Program is optimized for less then  99998 given nodes. \n");

				printf("No specific parameters needed. So, start the app like usual.\n");
				printf("\n");
				printf("\n");
				terminate = 1;
				break;
			}
			else
			{
				terminate = 1;
			}
		}
	}
	return terminate;
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
	exit(0);
}