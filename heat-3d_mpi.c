/* Include benchmark-specific header. */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#define m_printf if (myrank==0)printf
#define TSTEPS 40
#define N 20

double A[N][N][N];
double B[N][N][N];
int i, j, k, t;

int main(int argc, char** argv)
{
	int n = N;
	MPI_Request requests[4];
    MPI_Status statuses[4];
	
	int myrank, ranksize;
    int start_row, last_row, nrow;
    double start, end, time;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);
	
	start_row = (myrank * n) / ranksize;
    last_row = (((myrank + 1) * n) / ranksize) - 1;
	nrow = last_row - start_row + 1;
	
	for (i = start_row; i <= last_row; i++)
		for (j = 0; j < n; j++)
			for (k = 0; k < n; k++)
				A[i][j][k] = B[i][j][k] = (double) (i + j + n - k) * 10 / (n);
	
	start = MPI_Wtime();
	for (t = 1; t <= TSTEPS; t++) 
	{
		if (myrank) {
            MPI_Irecv(A[start_row - 1], n*n, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, requests);
        }
        if (myrank != ranksize - 1) {
            MPI_Isend(A[last_row], n*n, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, requests + 2);
        }
		if (myrank != ranksize - 1) {
            MPI_Irecv(A[last_row + 1], n * n, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, requests + 3);
        }
        if (myrank) {
            MPI_Isend(A[start_row], n * n, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, requests + 1);
        }
		int count = 4, shift = 0;
        if (!myrank) {
            count -= 2;
            shift = 2;
        }
        if (myrank == ranksize - 1) {
            count -= 2;
        }
        MPI_Waitall(count, requests + shift, statuses);
		for (i = start_row; i <= last_row; i++) {
			if((!i) || (i == n-1)){
				continue;
			}
			for (j = 1; j < n-1; j++) {
				for (k = 1; k < n-1; k++) {
					B[i][j][k] = 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
								 + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
								 + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
								 + A[i][j][k];
				}
			}
		}
		
		
		if (myrank) {
            MPI_Irecv(B[start_row - 1], n*n, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, requests);
        }
        if (myrank != ranksize - 1) {
            MPI_Isend(B[last_row], n*n, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, requests + 2);
        }
		if (myrank != ranksize - 1) {
            MPI_Irecv(B[last_row + 1], n * n, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, requests + 3);
        }
        if (myrank) {
            MPI_Isend(B[start_row], n * n, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, requests + 1);
        }
		count = 4, shift = 0;
        if (!myrank) {
            count -= 2;
            shift = 2;
        }
        if (myrank == ranksize - 1) {
            count -= 2;
        }
        MPI_Waitall(count, requests + shift, statuses);		
		for (i = start_row; i <= last_row; i++) {
			if((!i) || (i == n-1)){
				continue;
			}
			for (j = 1; j < n-1; j++) {
				for (k = 1; k < n-1; k++) {
				   A[i][j][k] = 0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
								+ 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
								+ 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
								+ B[i][j][k];
			   }
			}
		}
	}
	m_printf("Time of task = %f\n", MPI_Wtime() - start);
	return 0;
}
