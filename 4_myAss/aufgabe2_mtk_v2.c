#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include <iostream>
//using namespace std;

int main(int argc, char **argv)
{
    int x = 0;
    double kk;
    int proces;
    int numprocs;
    int right_neigh, left_neigh, up_neigh, down_neigh;
    int tag = 99;

    int n = 6; //size of matrices

    MPI_Status statRecv[2];
    MPI_Request reqSend[2], reqRecv[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proces);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);


    int sizeArray= n/numprocs;

    int *A[n], *B[n];
    for (int w1=0; w1<n; w1++){
    	A[w1]=(int *)malloc(n * sizeof(int));
    	B[w1]=(int *)malloc(n * sizeof(int));
    }
    for (int i = 0; i <  n; i++){
    	for (int j = 0; j < n; j++){
    		A[i][j]=0;
    		B[i][j]=0;
    	}
    }
    int *psa[sizeArray];
    int *psb[sizeArray];
    int *pra[sizeArray];
    int *prb[sizeArray];

           for (int i=0; i<	sizeArray; i++){
                   psa[i] = (int *)malloc(sizeArray * sizeof(int));
                   psb[i] = (int *)malloc(sizeArray * sizeof(int));
                   pra[i] = (int *)malloc(sizeArray * sizeof(int));
                   prb[i] = (int *)malloc(sizeArray * sizeof(int));
                   // Note that psa[i][j] is same as *(*(arr+i)+j)
           }
           for (int i = 0; i <  sizeArray; i++){
                 for (int j = 0; j < sizeArray; j++){
               	psa[i][j]=0;
               	psb[i][j]=0;
               	pra[i][j]=0;
               	prb[i][j]=0;
                   }
           }


        for (int i = 0; i < sizeArray; i++) { //let's make fist matrix
            for (int j = 0; j < sizeArray; j++) {
                psa[i][j] = 0;
                psb[i][j] = 0;

            }
        }

        for (int i = 0; i < sizeArray; i++) { //and the 2nd one
            for (int j = 0; j < sizeArray; j++) {
                pra[i][j] = psa[i][j];
                prb[i][j] = psb[i][j];

            }
        }

//    int *psa [n/numprocs][n/numprocs]; //nxn
//    int psb[n/numprocs][n/numprocs];
//    int pra[n/numprocs][n/numprocs];
//    int prb[n/numprocs][n/numprocs];
    int *c[n];
    for (int i=0; i<n; i++){
    	c[i]= (int *)malloc(n * sizeof(int));
    }
    for (int i = 0; i <  n; i++){
          for (int j = 0; j < n; j++){
          c[i][j]=0;
          }
    }

    int PP = numprocs;
    for (int i = 0; i < sizeArray / PP; i++)    {
        for (int j = 0; j < sizeArray / PP; j++)
        {
            psa[i][j] = A[proces / PP*(n / PP) + i][proces%PP*(n / PP) + j];
            psb[i][j] = B[proces / PP*(n / PP) + i][proces%PP*(n / PP) + j];
        }
    }



    double np = numprocs;
    kk = sqrt(np);
    int k = (int)kk;
    if (proces < k) // below neighbour set
    {
        left_neigh = (proces + k - 1) % k;
        right_neigh = (proces + k + 1) % k;
        up_neigh = ((k - 1)*k) + proces;
    }
    if (proces == k)
    {
        left_neigh = ((proces + k - 1) % k) + k;
        right_neigh = ((proces + k + 1) % k) + k;
        up_neigh = proces - k;
    }
    if (proces > k)
    {
        x = proces / k;
        left_neigh = ((proces + k - 1) % k) + x * k;
        right_neigh = ((proces + k + 1) % k) + x * k;
        up_neigh = proces - k;
    }
    if (proces == 0 || (proces / k) < (k - 1))
    {
        down_neigh = proces + k;
    }
    if ((proces / k) == (k - 1))
    {
        down_neigh = proces - ((k - 1)*k);
    }
    x = 0;
    int p = 0;
    do{
        if (p < proces / PP)
        {

                MPI_Irecv(pra, n*n / PP / PP, MPI_INT, right_neigh, tag, MPI_COMM_WORLD, &reqRecv[2]);
                MPI_Isend(psa, n*n / PP / PP, MPI_INT, left_neigh, tag, MPI_COMM_WORLD, &reqSend[2]);
                MPI_Wait(&reqRecv[2], &statRecv[2]);
                for (int i = 0; i < sizeArray / PP; i++)
                {
                    for (int j = 0; j < sizeArray / PP; j++)
                    {
                        psa[i][j] = pra[i][j];

                    }
                }
        }
            MPI_Barrier(MPI_COMM_WORLD);
        if (p < proces % PP)//
        {

            MPI_Irecv(prb, n*n / PP / PP, MPI_INT, down_neigh, tag, MPI_COMM_WORLD, &reqRecv[2]);
            MPI_Isend(psb, n*n / PP / PP, MPI_INT, up_neigh, tag, MPI_COMM_WORLD, &reqSend[2]);
            MPI_Wait(&reqRecv[2], &statRecv[2]);
            for (int i = 0; i < sizeArray / PP; i++)
            {
                for (int j = 0; j < sizeArray / PP; j++)
                {
                    psb[i][j] = prb[i][j];
                }
            }

        }
        MPI_Barrier(MPI_COMM_WORLD);
        p++;
    } while (p < n);

    for(int kk = 0; kk < PP; kk++) //algorithm
    {
        for (int i = 0; i < n / PP; i++)
        {
            for (int j = 0; j < n / PP; j++)
            {
                for (int k = 0; k < n / PP; k++)
                {
                    c[i][j] += psa[i][k] * psb[k][j];
                }
            }
        }
        MPI_Irecv(pra, n*n / PP / PP,MPI_INT,left_neigh, tag,MPI_COMM_WORLD, reqRecv);
        MPI_Irecv(prb, n*n / PP / PP,MPI_INT,down_neigh,tag,MPI_COMM_WORLD,&reqRecv[1]);
        MPI_Isend(psa, n*n / PP / PP,MPI_INT,right_neigh,tag,MPI_COMM_WORLD, reqSend);
        MPI_Isend(psb, n*n / PP / PP,MPI_INT,up_neigh,tag,MPI_COMM_WORLD,&reqSend[1]);
        MPI_Wait(reqRecv, statRecv);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < sizeArray / PP; i++)
           {
               for (int j = 0; j < sizeArray / PP; j++)
               {
                   psa[i][j] = pra[i][j];
               }
           }


           for (int i = 0; i < sizeArray / PP; i++)
           {
               for (int j = 0; j < sizeArray / PP; j++)
               {
                   psb[i][j] = prb[i][j];
               }
           }
    }

    printf("A");//show result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf(" ", pra[i][j]);
        }

    }
    printf("B");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
        	printf(" ", prb[i][j]);
        }
        ;
    }
    printf("C");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
        	printf(" ", c[i][j]);
        }

    }


    MPI_Finalize();

    return 0;
}
