/***********************************************************************
 Program: hellomp.c                                                  
 Author: Michael Czaja, Muttaki Aslanparcasi                                           
 matriclenumber: 4293033, 5318807                                             
 Assignment : 1                                                      
 Task: 3                                                             
 Parameters: no                                                      
 Environment variables: no                                           
                                                                     
 Description:                                                        
In this Assignment there is a method for numerical Integration. Here we use the Romberg Method combinded with MPI parallel programming.
/************************************************************************/


#include "mpi.h"	// import of the MPI definitions
#include <stdio.h>  // import of the definitions of the C IO library
#include <string.h> // import of the definitions of the string operations
#include <unistd.h> // standard unix io library definitions and declarations
#include <errno.h>  // system error numbers
#include <math.h>
#include <stdlib.h>



#define MAX_BUFFER_SIZE 1000

// TASK
double T(double x);
double F(double x);
double G(double x);
double P(double x);
double romberg(double f(double), double a, double b, int n, double **R);

int taskIspowerof2(unsigned int x);
int taskCalcHowManyRound(int sizeOfWorld);


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
	double idleTime=0;
	int steps = -1;
	int namelen;							 // length of name
	int my_rank;							 // rank of the process
	
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
	if (my_rank==0) {
		printf("Gib die Grenze A ein (double):\n");
		scanf("%lf",&globalBoundA);
		printf("Gib die Grenze B ein (double):\n");
		scanf("%lf",&globalBoundB);
		printf("Wieviele steps(>=1)? (int):\n");
		scanf("%d",&steps);
		printf("Bounds: %f > %f | steps (%d)\n",globalBoundA,globalBoundB,steps );
	} 	
	MPI_Bcast(&globalBoundA,1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&globalBoundB,1, MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&steps,1, MPI_INT, 0, MPI_COMM_WORLD);

	// ----------------------------------------------------------[ Para check ]--
	if(taskIspowerof2(world_size)!=1){
		utilOTPrint(my_rank,0, "ERROR\n");
		utilOTPrint(my_rank,0, "No, no, no --- Parameter n is not a power of 2\n");
		if(my_rank==0)
			utilPrintHelp();
		exit(0);
	}
	if(steps<1){
		utilOTPrint(my_rank,0, "ERROR\n");
		utilOTPrint(my_rank,0, "No, no, no --- parameter <steps> need to be >=1\n");
		if(my_rank==0)
			utilPrintHelp();
		exit(0);
	}
	

	

	// CALC
	int rounds= (log(world_size)/log(2));
	double diffAandB= globalBoundB-globalBoundA;
	double chunkSizeOfPro = diffAandB / world_size;

	//start
	double myA = globalBoundA+( chunkSizeOfPro * my_rank);
	double myB = globalBoundA+( chunkSizeOfPro * (my_rank +1));

	printf("Me (%d) will calc from (%f) to (%f)\n",my_rank,myA,myB);
	double resultFunction1 = 100.0;
	double resultFunction2 = 100.0;	

	double F(double), G(double), P(double);
	
	n = steps;
	R = calloc((n + 1), sizeof(double *));
	for (i = 0; i <= n; i++)
		R[i] = calloc((n + 1), sizeof(double));

	resultFunction1 = romberg(G, myA, myB, n, R); 
	resultFunction2 = romberg(P, myA, myB, n, R); 

	// ------------------------------------------------------------[Para Part]--
	int dataToSend=2;

	double mpi_loopStart = MPI_Wtime();
	for(int round = rounds-1; round>=0; round--){
	    int second = (1 << round);
	    int partnerRank= my_rank ^ second;
		double dataArrayToSend[2]= {resultFunction1,resultFunction2};
		double dataArrayToRev[2];
        
		int flag = 0;
		MPI_Request request, request2;
    	MPI_Status status;
		
		printf("ROUND(%d) Me(%d) call Partner(%d)\n", round, my_rank, partnerRank);
	
		/*
		MPI_Sendrecv(&dataArrayToSend, dataToSend, MPI_DOUBLE, partnerRank, round, &dataArrayToRev, dataToSend, MPI_DOUBLE, partnerRank, round, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		*/

		if(my_rank<partnerRank){
		MPI_Isend(dataArrayToSend, dataToSend, MPI_DOUBLE, partnerRank, 123, MPI_COMM_WORLD, &request2);
		MPI_Irecv(dataArrayToRev, dataToSend, MPI_DOUBLE, partnerRank, 123, MPI_COMM_WORLD, &request);
		}
		else{
		MPI_Irecv(dataArrayToRev, dataToSend, MPI_DOUBLE, partnerRank, 123, MPI_COMM_WORLD, &request);
		MPI_Isend(dataArrayToSend, dataToSend, MPI_DOUBLE, partnerRank, 123, MPI_COMM_WORLD, &request2);
		}

		 MPI_Test(&request, &flag, &status);
		 
		 double mpi_ideleTimeStart = MPI_Wtime();
   		 while (!flag)
    	{
			idleOperations++;
        // order pizza, sandwiches or ice -cream ..:) 
        MPI_Test(&request, &flag, &status);
    	}
		double mpi_idleTimeEnd = MPI_Wtime();
    	idleTime += mpi_idleTimeEnd-mpi_ideleTimeStart;
		/*
		// Use this if you want to wait ;)
		MPI_Wait(&request, &status);
    	MPI_Wait(&request2, &status);
		*/
		resultFunction1 += dataArrayToRev[0];
		resultFunction2 += dataArrayToRev[1];
	}
	double mpi_loopEnd = MPI_Wtime();


	// --------------------------------------------------------[Para Part END]--
	double mpi_programEnd = MPI_Wtime();
	double loopTimaAtAll= mpi_loopEnd-mpi_loopStart;
	
	// ----------------------------------------------------------[Result Call]--
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(100);
	if(my_rank==0){
	utilOTPrint(0, my_rank,"--------------------- RESULT\n");
	printf("boundaries: %f -> %f  | steps = %d \n", globalBoundA, globalBoundB, n);
	utilOTPrint(0, my_rank,"A:=log(7 * x) / x       | B:=sqrt((3 * x) + 2)\n");
	printf("(A)=%f (B)=%f  \n", resultFunction1, resultFunction2);
	printf("\n");
	utilOTPrint(0, my_rank,"--------------------- MES\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(100);
	printf("[Node %d] Ia=%f Ie=%f steps=%d (A)=%f (B)=%f  loopTime(%f) IdleOP's(%f) IdleTime(%f)\n", my_rank, myA,myB, n ,resultFunction1, resultFunction2, loopTimaAtAll, idleOperations, idleTime);
	MPI_Buffer_detach((void *)buffer, &bsize);
	MPI_Finalize(); // finalizing MPI interface
    return 0;
}


// ------------------------------------------------------------------[ TaskFunc. ]--


/**
 * @brief Checks if a number is a power of 2.
 * 
 * @param x Number
 * @return int 1 if so.
 */
int taskIspowerof2(unsigned int x)
{
	return x && !(x & (x - 1));
}

/**
 * @brief Calcs the communications-rounds.
 * 
 * @param sizeOfWorld Size of the world(# nodes).
 * @return int Number of rounds to communicate.
 */
int taskCalcHowManyRound(int sizeOfWorld)
{
	int rounds = log(sizeOfWorld) / log(2);
	return rounds;
}



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






double F(double x) {
	return (1.0 / (1.0 + x));
}

double T(double x) {
	return 1.0;
}

double G(double x) {
	return (log(7 * x) / x);
}

double P(double x) {
	double result = sqrt((3 * x) + 2);
	return result;
}

double romberg(double f(double), double a, double b, int n, double **R) //Hier MPI_Bcast fuer die Grenzen l(a)-r(b)
{	
	// TODO Hier ein bisschen aufräumen, raus was wir nicht brauchen. Die kommentarzeilen und so
	int i, j, k;
	double h, sum;
	h = b - a;
	R[1][1] = 0.5 * h * (f(a) + f(b));
	//printf("(%d) Rs[1][1] = %f\n", my_rank, R[1][1]);
	for (i = 2; i <= n; i++) {
		h = (b-a)*pow(2,1-i);
//		h *= 0.5;
		sum = 0;
		for (k = 1; k <= pow(2, i-2); k += 1) {
			sum += f(a + ((2*k)-1) * h);
		}
		R[i][1] = 0.5 * R[i - 1][1] + sum * h; //Inidizes vertauscht?!
		for (j = 1; j <= i; j++) {
			R[i][j+1] = R[i][j] + (R[i][j] - R[i - 1][j])
					/ (pow(4, j) - 1);
			//printf("Ende -> R[%d][%d] = %f\n", i, j, R[i][j]);
		}
		//printf("{ Give That to other process %d / %d} R[%d][0] = %1f\n", i, n,
		//		i, R[i][1]);
	}
	//printf("My result for integral: R[%d][%d] -- %f\n\n", n, n, R[n][n]);
	double toReturn = R[n][n];
	return toReturn;
	/*int ns = 10;
	double myA = 0;
	double myB = 1;
	double **T;
	for (int var = 0; var < ns; ++var) {
		for (int b = 0; b < ns; ++b) {
			R[var][b] = 0;
		}
	}

	T = calloc((ns + 1), sizeof(double *));
	for (i = 0; i <= ns; i++)
		T[i] = calloc((ns + 1), sizeof(double));
	for (int myI = 0; myI < ns; ++myI) {
		for (int myJ = 0; myJ < ns; ++myJ) {
			printf("T[%d][%d] -- %f\n", myI, myJ, T[myI, myJ]);
		}
	}
	*/

}
