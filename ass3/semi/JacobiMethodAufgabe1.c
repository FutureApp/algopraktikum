/************************************************************************/
/* Program: JacobiMethodAufgabe1.c										*/
/* Group: pa1807								                        */ 
/* Author: Kerem Bozyel kerem.bozyel@live.de							*/
/*	   Mansoor Stuman m.stuman@yahoo.de									*/
/*	   Günay Bayraktar g_bayraktar@web.de								*/
/* Matriclenumber: Kerem 5712982										*/
/*		   Mansoor 5749175												*/
/*		   Günay 5352614                                        		*/
/* Assignment : 3                                                       */	
/* Task: 1                                                              */
/* Parameters: no                                                       */
/* Environment variables: no                                            */
/*                                                                      */
/* Description:                                                         */
/*                                                                      */
/* Solving a linear system of equations with the Jacobi Method	        */
/* Row oriented													        */
/************************************************************************/


#include "mpi.h" 	    // import of the MPI definitions
#include <stdio.h> 	    // import of the definitions of the C IO library
#include <string.h>     // import of the definitions of the string operations
#include <unistd.h>	    // standard unix io library definitions and declarations
#include <errno.h>	    // system error numbers
#include <time.h>
#include <stdlib.h>
#include <math.h>

/****************************************************/
/* Name: butterflyAlgorithm		 					*/
/* 													*/
/* Parameter: 										*/
/* Name: n -->  Anzahl Prozesse						*/
/* Return value: no									*/
/* 													*/
/* Function description:							*/
/* Die Summe wird über eine Butterfly				*/
/* Struktur aus allen Teilsummen zusammengefügt		*/
/* 													*/
/****************************************************/
void butterflyAlgorithm(int my_rank, double *sum, int n){
	double buf = 0.0;
	int tag = 0;
	MPI_Status status;				// Enthält entweder eine Errormeldung oder Success
	MPI_Request request;
	
	for(int i = log2(n) - 1; i >= 0; i--){
		//Differenz zweier Potenzen ist eine Zweierpotenz und XOR Operator einfach zu identifizieren
		//Beispiel: n = 8; Prozess 1 muss Prozess 5 senden
		//Binär: 1 = 0001 und 5 = 0101 --> XOR ergibt dann: 0100 und das ist Abstand 4 ((int) pow(2,i))
		//Wir "xoren" den aktuellen Rang mit dem Abstand und erhalten dann den Nachbarn
		int neighbour = my_rank ^ ((int) pow(2,i));
		//Non-blocking communication
		//Sende an Nachbarn
		MPI_Isend(sum,1,MPI_DOUBLE,neighbour,tag,MPI_COMM_WORLD,&request);
		//Erhalte von Nachbarn
		MPI_Recv(&buf,1,MPI_DOUBLE,neighbour,tag,MPI_COMM_WORLD,&status);
		//Neuberechnung
		(*sum) += buf;
	}
}

int main(int argc, char* argv[ ])
{ 
	int my_rank;					// rank of the process
	int n; 							// n = number of processes
	int tag = 0;						// Tag der Kommunikation
	MPI_Status status;				// Enthält entweder eine Errormeldung oder Success
	MPI_Request request;
	MPI_Init(&argc, &argv);		 	// initializing of MPI-Interface
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	//get your rank
	MPI_Comm_size (MPI_COMM_WORLD, &n); //find out how many processes are started
	
	//Dient für die Verifizierung, ob der User "-h" eingegeben hat
	_Bool isFlagEntered = 0;
	//Es werden alle Parameter von der Eingabe in der Console durchlaufen 
	for(int i = 0; i < argc; i++){
		//Falls einer der Parameter ein "-h" ist
		//strcmp(string x,string y) gibt 0, wenn x und y gleiche Strings sind; Ansonsten 1 
		if(strcmp(argv[i], "-h") == 0){
			//So wird der Hilfemodus auf true gesetzt
			isFlagEntered = 1;
			//Damit müssen weitere Parameter nicht weiterdurchlaufen werden
			break;
		}
	}
	
	//Falls zu wenig Parameter und kein -h eingegeben wurde, dann wird das Programm beendet.
	if(argc <= 3 && (isFlagEntered == 0)){
		printf("To few parameters\n");
		MPI_Finalize();
		return 0;
	}
	
	if(isFlagEntered == 1 && my_rank == 0){//Hilfemodus aktiviert
		//Hilfetexte werden geprintet
		printf("First: Compile your MPI Programm with:\n");
		printf("mpicc -o <program_name> <source_program_name>.c\n");
		printf("<source_program_name>.c is your Code written in C\n");
		printf("<program_name> the executable programm of your written Code. You have to give a name\n");
		printf("\n");
		printf("Second execute your program with:\n");
		printf("mpiexec -np <number_of_processes> -f <hostfile> ./<program_name> ./<matrixAFile> ./<vectorBFile>\n");
		printf("<number_of_processes> the number of process will start the programm\n");
		printf("<matrixAFile> the matrix A filename\n");
		printf("<vectorBFile> the vector b filename\n");
		printf("-f <hostfile> hostfile to execute on more computer\n");
		MPI_Finalize();
		return 0;
	}else{
		char* filenameForMatrixA = "";
		char* filenameForVectorB = "";
		char* filenameForVectorX = "Vector_x_Aufgabe1";
		double epsilon = 0.0;
		
		//Schritt 1: Lese MatrixA Datei ein und übergebe Werte
		//Übergebe Parameter
		filenameForMatrixA = argv[1];
		filenameForVectorB = argv[2];
		epsilon = atof(argv[3]); //atof konvertiert von char zu double
		
		int error = 0;
		MPI_File matrixAFile;
		MPI_File vectorBFile;
		MPI_File vectorXFile;
		
		//Öffne MatrixA Datei
		error = MPI_File_open ( MPI_COMM_SELF, filenameForMatrixA, MPI_MODE_RDONLY, MPI_INFO_NULL, &matrixAFile);
		
		//Falls Datei nicht gelesen werden konnte, dann abbrechen
		if ( error == 1 ) {
			MPI_Abort ( MPI_COMM_WORLD, 1 );
		};
		
		MPI_Offset fileSizeMatrix;
		MPI_File_get_size(matrixAFile,&fileSizeMatrix);

		double size = fileSizeMatrix / sizeof(double);
		int numOfElem = size / n;
		int lines = sqrt(size);

		MPI_File fhandle;
		MPI_Offset fsize;

		int err = 0;
		err = MPI_File_open(MPI_COMM_WORLD, pathToMatrix, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle);
		err = MPI_File_get_size(fhandle, &fsize);

		int dimOfMatrix = sqrt(fileSizeMatrix / sizeof(double));
		int blocksToHandle = dimOfMatrix / n;
		int elemsToHandle = dimOfMatrix * blocksToHandle;

		//Lesen der Werte
		int linesForProc = numOfElem / lines;
		
		double matrixBuffer[elemsToHandle];
		//error = MPI_File_read_ordered_begin( matrixAFile, matrixBuffer, numOfElem*lines, MPI_DOUBLE);
		MPI_File_read_ordered(matrixAFile, matrixBuffer, blocksToHandle, MPI_DOUBLE, MPI_STATUS_IGNORE);
		//Beenden der I/O Operation
		//error = MPI_File_read_ordered_end( matrixAFile, matrixBuffer, &status );
		//Schließe MatrixA Datei
		MPI_File_close(&matrixAFile);
		//Werte von Buffer an Matrix übergeben

		for (int x = 0; x < n; x++)
		{
			if (my_rank == x)
			{	printf("My Rank - %d\n", my_rank);
				for (int i = 0; i < numOfElem; i++)
				{
					if (i % lines == 0)
					{
						printf("\n");
					}
					printf("%lf ", matrixBuffer[i]);
				}
			}
		}
		printf("\n\n");

		/*
		double test[lines * lines];
		memset(test, 0, sizeof(test)); //Mit 0en füllen
		
		for(int i = 0; i < numOfElem; i++){
			test[my_rank*numOfElem + i] = matrixBuffer[i];
		}*/
		
		
		
		double matrixA[linesForProc][lines];
		memset(matrixA, 0, sizeof(matrixA)); //Mit 0en füllen
		
		//Matrix mit Buffer befüllen
		for(int i = 0; i < numOfElem; i++){
			matrixA[i / lines ][i % lines] = matrixBuffer[i];
		}
		
		//Schritt 2: Überprüfe, ob die Matrix "diagonaldominant" ist
		for (int i = 0; i < linesForProc; i++) {
			double sum = 0.0;
			for (int j = 0; j < lines; j++) {
				if(i != j){//Diagonalelemente werden nicht mit summiert
					sum += matrixA[i][j];
				}
			}
			//Falls die Summe negativ ist, dann mit -1 multiplizieren, um den Betrag zu bekommen
			double absSum = (sum < 0) ? sum*(-1) : sum; 
			//Falls das Diagonalelement negativ ist, dann mit -1 multiplizieren, um den Betrag zu bekommen
			double absDiag = (matrixA[i][i] < 0) ? matrixA[i][i]*(-1) : matrixA[i][i];
			if(absDiag <= absSum){
				if(my_rank == 0) printf("Matrix A is not diagonaldominant\n");
				MPI_Finalize();
				return 0;
			}	
		}
		
		//Schritt 3: Lese VectorB Datei ein und übergebe Werte
		error = MPI_File_open ( MPI_COMM_SELF, filenameForVectorB, MPI_MODE_RDONLY, MPI_INFO_NULL, &vectorBFile);
		
		if ( error == 1 ) {
			MPI_Abort ( MPI_COMM_WORLD, 1 );
		};
		
		MPI_Offset fileSizeVector;
		MPI_File_get_size(vectorBFile,&fileSizeVector);
		int vecElem = fileSizeVector / sizeof(double);
		double vectorB[vecElem];
		//Lesen der Werte
		error = MPI_File_read(vectorBFile, vectorB, vecElem, MPI_DOUBLE, &status);
		MPI_File_close(&vectorBFile);
		
		//Höhe von Vektor muss gleich sein mit Breite der Matrix, sonst kann man nicht multiplizieren
		//Da Matrix quadratisch ist, kann man die Variable lines verwenden
		if(vecElem != lines){
			if(my_rank == 0) printf("Height of VectorB is not equal with width of Matrix\n");
			MPI_Finalize();
			return 0;
		}
		
		//Schritt 4: Berechnungen mithilfe der Jacobi Methode ausführen
		double xPrev[vecElem];	//die x-Werte in der vorherigen Iteration
		double xCurr[vecElem];	//aktuelle x-Werte
		
		memset(xPrev, 0, sizeof(xPrev));
		memset(xCurr, 0, sizeof(xCurr));
		
		
		_Bool distanceIsSmallEnough = 1;//TODO
		while(distanceIsSmallEnough == 0){//Solange die Bedingung nicht klein genug ist, werden alle Berechnungen weitergeführt
			for(int i = 0; i < vecElem; i++){
				double firstSum = 0.0;
				double secondSum = 0.0;
				//TODO Indizes müssen angepasst werden (wurde noch net getestet, ob aktuell klappt)
				for(int j = 0; j < i; j++){
					firstSum += matrixA[i][j]*xPrev[j];
				}
				//TODO Indizes müssen angepasst werden (wurde noch net getestet, ob aktuell klappt)
				for(int j = i; j < lines; j++){
					secondSum += matrixA[i][j]*xPrev[j];
				}
				if(my_rank == 0) printf("First Sum : %lf\n", firstSum);
				if(my_rank == 0) printf("Second Sum : %lf\n", secondSum);
				break;
				xCurr[i] = (1/matrixA[i][i]) * (vectorB[i] - firstSum - secondSum);
			}
			
			//Summieren der vorherigen und aktuellen x-Werte
			double dif = 0.0;
			double sum = 0.0;
			for(int i = 0; i < lines; i++){
				dif = (xCurr[i] - xPrev[i]);
				dif = (dif < 0) ? dif * (-1) : dif; //Betrag der Differenz bestimmen
				sum = sum + dif;
			}
			//Falls distance klein genug ist, dann sind wir fertig
			if(sum < epsilon){
				distanceIsSmallEnough = 1;
			}
		}
		
		//Schritt 5: Alles in Output Datei schreiben
		error = MPI_File_open( MPI_COMM_WORLD, filenameForVectorX, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &vectorXFile ) ; 
		MPI_File_write_ordered_begin(vectorXFile, xCurr, lines, MPI_DOUBLE);
		MPI_File_write_ordered_end(vectorXFile, xCurr,&status);
		//Test
		/*error = MPI_File_open ( MPI_COMM_SELF, filenameForVectorX, MPI_MODE_RDONLY, MPI_INFO_NULL, &vectorXFile);
		
		if ( error == 1 ) {
			MPI_Abort ( MPI_COMM_WORLD, 1 );
		};
		
		double testBuffer[lines];
		//Lesen der Werte
		error = MPI_File_read(vectorXFile, testBuffer, lines, MPI_DOUBLE, &status);
		MPI_File_close(&vectorBFile);
		for(int i = 0; i < lines; i++){
			printf("%lf ", testBuffer[i]);
		}*/
		
		MPI_Finalize();		            // finalizing MPI interface
	}
	return 0;						// end of progam with exit code 0 
} 


