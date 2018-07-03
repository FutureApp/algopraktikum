#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    int x = 0;
    double kk;
    int proces;
    int numprocs;
    int right_neigh, left_neigh, up_neigh, down_neigh;
    int tag = 99;

    static const int n = 6; //size of matrices

    int psa[n][n]; //nxn
    int psb[n][n];
    int pra[n][n];
    int prb[n][n];
    int c[n][n];

    for (int i = 0; i < n; i++) { //let's make fist matrix
        for (int j = 0; j < n; j++) {
            psa[i][j] = (int)rand() % 100 + 1;
            psb[i][j] = (int)rand() % 100 + 1;
            c[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++) { //an the 2nd one
        for (int j = 0; j < n; j++) {
            pra[i][j] = psa[i][j];
            prb[i][j] = psb[i][j];

        }
    }
    MPI_Status statRecv[2];
    MPI_Request reqSend[2], reqRecv[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proces);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);


    int PP = numprocs;
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
        MPI_Irecv(pra, n*n / PP / PP,MPI_FLOAT,left_neigh, tag,MPI_COMM_WORLD, reqRecv);
        MPI_Irecv(prb, n*n / PP / PP,MPI_FLOAT,down_neigh,tag,MPI_COMM_WORLD,&reqRecv[1]);
        MPI_Isend(psa, n*n / PP / PP,MPI_FLOAT,right_neigh,tag,MPI_COMM_WORLD, reqSend);
        MPI_Isend(psb, n*n / PP / PP,MPI_FLOAT,up_neigh,tag,MPI_COMM_WORLD,&reqSend[1]);
        MPI_Wait(reqRecv, statRecv);

    }

    cout << "A" << endl; //show result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << pra[i][j] << " ";
        }
        cout << endl;
    }
    cout << "B" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << prb[i][j] << " ";
        }
        cout << endl;
    }
    cout << "C" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << c[i][j] << " ";
        }
        cout << endl;
    }


    MPI_Finalize();

    return 0;
}
