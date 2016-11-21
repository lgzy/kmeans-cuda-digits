#include "BMP.h"

#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <ctime>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <device_functions.h>
//#include <cuda_runtime_api.h>

const int D = 28 * 28;
const int CLUSTERS = 14;
const float threshold = 0.0;
const float eps = 1e-6;
const int tile = 1024;
const int clustersInShared = 14;


typedef unsigned char uchar;

using namespace std;

void read4(int * x, FILE * input) {
	uchar temp;
	*x = 0;
	for (int i = 0; i < 4; ++i) {
		fread(&temp, 1, 1, input);
		*x |= (temp << ((4 - i - 1) * 8));
	}
}

void read(int * nObj, float ** obj, uchar ** membership, string filename = "") {
	FILE * input;
	assert(input = fopen((filename + "images").c_str(), "rb"));
	int magic, row, column;
	read4(&magic, input);
	read4(nObj, input);
	read4(&row, input);
	read4(&column, input);

	//assert(*nObj == 60000);
	assert(row == 28);
	assert(column == 28);
	int size = (*nObj) * D;

	printf(" Number of objects = %d\n row = %d\n column = %d\n", *nObj, row, column);
	uchar * charObj;
	void * temp = malloc(size);	assert(temp);
	charObj = (uchar *) temp;
	//printf("size = %d\n", size);
	assert(fread(charObj, 1, size, input) == size);
	*obj = new float[size];
	for (int i = 0; i < size; ++i){
		(*obj)[i] = charObj[i];
	}
	for (int i = 0; i < *nObj; ++i) {
		for (int row = 0; row < 28 / 2; ++row) {
			for (int col = 0; col < 28; ++col) {
				swap((*obj)[i * D + row * 28 + col], (*obj)[i * D + (28 - row - 1) * 28 + col]);
			}
		}
	}

	free(charObj);
	fclose(input);

	assert(input = fopen((filename + "labels").c_str(), "rb"));
	read4(&magic, input);
	read4(nObj, input);
	//assert(*nObj == 60000);
	*membership = new uchar[*nObj];
	fread(*membership, *nObj, 1, input);
	
	//for (int i = 0; i < 10; ++i)
	//	printf("label of the %d-th = %u\n", i, (*membership)[i]);

	fclose(input);

	puts("READ SUCCESSFUL");
}

__host__ __device__ inline float dist(float * v1, float * v2) {
	float res = 0;
	for (int i = 0; i < D; ++i) {
		//if (v2[i] < v1[i]) res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		res += (*(v2 + i) - *(v1 + i)) * (*(v2 + i) - *(v1 + i));
	}
	//printf("%.5f\n", sqrt(res));
	return res;
}

void print(FILE * file, float * v) {
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			fprintf(file, "%0.f ", v[i * 28 + j]);
		}
		fprintf(file, "\n");
	}
}

void writeBMP(float * img, string filename) {
	FILE * tar = fopen(filename.c_str(), "wb");
	char temp = 'B';
	fwrite(&temp, 1, 1, tar);
	temp = 'M';
	fwrite(&temp, 1, 1, tar);


	int sizeBMP = 26 * 26 * 3 + 4 + 4 + 4 + 2 + sizeof(DIBHeader);

	fwrite(&sizeBMP, 4, 1, tar);
	fwrite(&sizeBMP, 4, 1, tar);

	int offset = 2 + 4 + 4 + 4 + sizeof(DIBHeader);
	fwrite(&offset, 4, 1, tar);

	DIBHeader dib(28, 28);
	fwrite(&dib, sizeof(DIBHeader), 1, tar);

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			unsigned char x = img[i * 28 + j] + 0.5;
			for (int k = 0; k < 3; ++k) {
				fwrite(&x, 1, 1, tar);
			}
		}
	}
	fclose(tar);
}

void sequential(float * obj, int nObj, float * clusters, uchar * membership) {
	int alpha;
	int loops = 0;
	float * sum = new float[CLUSTERS * D];
	int * members = new int[CLUSTERS];
	do{
		for (int i = 0; i < CLUSTERS; ++i) {
			members[i] = 0;
			for (int k = 0; k < D; ++k) {
				sum[i * D + k] = 0;
			}
		}
		alpha = 0;
		for (int i = 0; i < nObj; ++i) {
			int minid = membership[i];
			float mind = dist(obj + i * D, clusters + minid * D);
			for (int j = 0; j < CLUSTERS; ++j) {
				float x = dist(obj + i * D, clusters + j * D);
				if (x - mind < -eps) {
					mind = x;
					minid = j;
				}
			}
			if (membership[i] != minid) {
				membership[i] = minid;
				alpha++;
			}
			members[membership[i]] ++;
			for (int k = 0; k < D; ++k) {
				sum[membership[i] * D + k] += obj[i * D + k];
			}
		}
		for (int i = 0; i < CLUSTERS; ++i) {
			for (int k = 0; k < D; ++k) {
				clusters[i * D + k] = sum[i * D + k] * 1. / members[i];
			}
		}
		++loops;
		printf("%d -> %.5f\n", loops, 1. * alpha / nObj);
	} while (1. * alpha / nObj > threshold && loops < 500);
}

void check(uchar * actual, uchar * proposed, float * clusters, int nObj, int nTest, float *testObj, uchar * testActual) {
	int * count = new int[CLUSTERS * CLUSTERS];
	int * represents = new int[CLUSTERS];
	memset(count, 0, CLUSTERS * CLUSTERS * sizeof(int));
	for (int i = 0; i < nObj; ++i) {
		count[proposed[i] * CLUSTERS + actual[i]]++;
	}
	for (int i = 0; i < CLUSTERS; ++i) {
		int mx = -1;
		for (int j = 0; j < CLUSTERS; ++j) {
			if (mx == -1 || count[i * CLUSTERS + j] >  count[i * CLUSTERS + mx]) {
				mx = j;
			}
		}
		represents[i] = mx;
	}
	/*
	for (int i = 0; i < CLUSTERS; ++i) {
		for (int j = 0; j < CLUSTERS; ++j) {
			printf("%d\t", count[i * CLUSTERS + j]);
		}
		puts("");
	}
	*/

	
	for (int i = 0; i < CLUSTERS; ++i) {
		cout << i << " represents " << represents[i] << endl;
	}	

	int wrong = 0;
	for (int i = 0; i < nObj; ++i) {
		wrong += actual[i] != represents[proposed[i]];
	}
	puts("----- On training -----");
	printf("wrong %d out of %d\n", wrong, nObj);
	printf("in percentage %.2f\n", 1. * wrong / nObj);

	puts("----- On test ----");
	wrong = 0;
	for (int i = 0; i < nTest; ++i) {
		int mycluster = -1;
		float mind;
		for (int j = 0; j < CLUSTERS; ++j) {
			if (mycluster == -1 || dist(clusters + j * D, testObj + i * D) < mind) {
				mind = dist(clusters + j * D, testObj + i * D);
				mycluster = j;
			}
		}
		wrong += testActual[i] != represents[mycluster];
	}

	printf("wrong %d out of %d\n", wrong, nTest);
	printf("in percentage %.2f\n", 1. * wrong / nTest);
}

__global__ void simpleFindCluster(float * obj, int nObj, float * clusters, uchar * membership) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nObj) {
		int x = membership[id];
		float mind = dist(obj + id * D, clusters + x * D);
		for (int j = 0; j < CLUSTERS; ++j) {
			float candd = dist(obj + id * D, clusters + j * D);
			if (mind > candd) {
				mind = candd;
				x = j;
			}
		}
		membership[id] = x;
	}
}


void simpleParallel(float * obj, int nObj, float * clusters, uchar * membership) {
	int alpha = 0;
	int loops = 0;

	float * d_obj;
	float * d_clusters;
	uchar * d_membership, * temp;
	temp = new uchar[nObj];

	float * sum = new float[CLUSTERS * D];
	int * members = new int[CLUSTERS];


	cudaMalloc((void **)&d_obj, nObj * D * sizeof(float));
	cudaMalloc((void**)&d_clusters, CLUSTERS * D * sizeof(float));
	cudaMalloc((void**)&d_membership, nObj);
	cudaMemcpy(d_obj, obj, nObj * D * sizeof(float), cudaMemcpyHostToDevice); 	
	//cudaSuccess();

	dim3 blocksPerGrid((nObj + tile - 1) / tile);
	dim3 threadsPerBlock(tile);

	do{
		cudaMemcpy(d_clusters, clusters, CLUSTERS * D * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_membership, membership, nObj, cudaMemcpyHostToDevice);
		simpleFindCluster << <blocksPerGrid, threadsPerBlock >> > (d_obj, nObj, d_clusters, d_membership);
		cudaDeviceSynchronize();
		cudaMemcpy(temp, d_membership, nObj, cudaMemcpyDeviceToHost);

		//memset(sum, 0, CLUSTERS * D * sizeof(float));
		//memset(members, 0, CLUSTERS * sizeof(int));
		for (int i = 0; i < CLUSTERS; ++i) {
			members[i] = 0;
			for (int j = 0; j < D; ++j) {
				sum[i * D + j] = 0.0;
			}
		}
		
		alpha = 0;

		for (int i = 0; i < nObj; ++i) {
			alpha += temp[i] != membership[i];
			membership[i] = temp[i];
			members[membership[i]]++;
			for (int j = 0; j < D; ++j) {
				sum[membership[i] * D + j] += obj[i * D + j];				
			}
		}
		for (int i = 0; i < CLUSTERS; ++i) {
			for (int j = 0; j < D; ++j) {
				clusters[i * D + j] = sum[i * D + j] / members[i];
			}
		}
		++loops;
		printf("%d -> %.5f\n", loops, 1. * alpha / nObj);
	} while (1. * alpha / nObj > threshold && loops < 500);
}


__global__ void tiledFindCluster(float * obj, int nObj, float * clusters, uchar * membership) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	float self[D];
	for (int i = 0; i < D; ++i) {
		self[i] = (id < nObj) ? obj[id * D + i] : 0;
	}
	__shared__ float sharedClusters[clustersInShared * D];
	int x = (id < nObj) ? membership[id] : 0;
	float mind = dist(self, clusters + x * D);
	float candd;
	int N;
	for (int k = 0; k < (CLUSTERS - 1) / clustersInShared + 1; ++k) {
		N = (CLUSTERS - k * clustersInShared < clustersInShared) ? CLUSTERS - k * clustersInShared : clustersInShared;

		for (int i = 0; i < D; ++i) {
			if(threadIdx.x < N) sharedClusters[threadIdx.x * D + i] = clusters[(k * clustersInShared + threadIdx.x) * D + i];
		}

		__syncthreads();					

		for (int j = 0; j < N; ++j) {

			candd = dist(self, sharedClusters + j * D);
			/*for (int f = 0; f < D; ++f) {
				candd += (self[f] - sharedClusters[j * D + f]) * (self[f] - sharedClusters[j * D + f]);
			}*/
			if (mind > candd) {
				mind = candd;
				x = k * clustersInShared + j;
			}
		}
			
		__syncthreads();
	}
	if(id < nObj) membership[id] = x;
}

void tiledParallel(float * obj, int nObj, float * clusters, uchar * membership) {
	int alpha = 0;
	int loops = 0;

	float * d_obj;
	float * d_clusters;
	uchar * d_membership, *temp;
	temp = new uchar[nObj];

	float * sum = new float[CLUSTERS * D];
	int * members = new int[CLUSTERS];


	cudaMalloc((void **)&d_obj, nObj * D * sizeof(float));
	cudaMalloc((void**)&d_clusters, CLUSTERS * D * sizeof(float));
	cudaMalloc((void**)&d_membership, nObj);
	cudaMemcpy(d_obj, obj, nObj * D * sizeof(float), cudaMemcpyHostToDevice);
	//cudaSuccess();

	dim3 blocksPerGrid((nObj + tile - 1) / tile);
	dim3 threadsPerBlock(tile);

	do{
		clock_t start = clock();
		cudaMemcpy(d_clusters, clusters, CLUSTERS * D * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_membership, membership, nObj, cudaMemcpyHostToDevice);
		tiledFindCluster << <blocksPerGrid, threadsPerBlock >> > (d_obj, nObj, d_clusters, d_membership);
		cudaDeviceSynchronize();
		cudaMemcpy(temp, d_membership, nObj, cudaMemcpyDeviceToHost);

		//printf(" finding = %.3f s   ", (clock() - start) * 1. / CLOCKS_PER_SEC);
		start = clock();
		//memset(sum, 0, CLUSTERS * D * sizeof(float));
		//memset(members, 0, CLUSTERS * sizeof(int));
		for (int i = 0; i < CLUSTERS; ++i) {
			members[i] = 0;
			for (int j = 0; j < D; ++j) {
				sum[i * D + j] = 0.0;
			}
		}

		alpha = 0;

		for (int i = 0; i < nObj; ++i) {
			alpha += temp[i] != membership[i];
			membership[i] = temp[i];
			members[membership[i]]++;
			for (int j = 0; j < D; ++j) {
				sum[membership[i] * D + j] += obj[i * D + j];
			}
		}
		for (int i = 0; i < CLUSTERS; ++i) {
			for (int j = 0; j < D; ++j) {
				clusters[i * D + j] = sum[i * D + j] / members[i];
			}
		}

		//printf("recalc = %.3f s\n", (clock() - start) * 1. / CLOCKS_PER_SEC);

		++loops;
		printf("membership change rate on %d-th iteration -> %.5f\n", loops, 1. * alpha / nObj);
	} while (1. * alpha / nObj > threshold && loops < 500);
}

int main() {
	srand(time(NULL));
	int nObj;
	float * obj;
	uchar * actual;

	int nTest;
	float * testObj;
	uchar * testActual;

	read(&nObj, &obj, &actual, "");
	read(&nTest, &testObj, &testActual, "test-");

	//need to assign clusters initially
	//TODO add kmeans++ assigning
	float * initialClusters = new float[D * CLUSTERS];

	int * temp = new int[nObj];

	for (int i = 0; i < nObj; ++i) {
		temp[i] = i;
	}
	random_shuffle(temp, temp + nObj);

	for (int i = 0; i < CLUSTERS; ++i) {
		int from = temp[i];
		memcpy(initialClusters + i * D, obj + from * D, D * sizeof(float));
	}
	clock_t start;


	nObj = 60000;
	/*----- Sequential--------*/
	{
		start = clock();
		float * clusters = new float[D * CLUSTERS];
		memcpy(clusters, initialClusters, D * CLUSTERS * sizeof(float));
		uchar * membership = new uchar[nObj];
		memset(membership, 0, nObj);

		sequential(obj, nObj, clusters, membership);

		check(actual, membership, clusters, nObj, nTest, testObj, testActual);
		printf("Sequential Total execution time = %.2f\n", (clock() - start) * 1. / CLOCKS_PER_SEC);
	}
	/*----- End Sequential-----*/




	/*------ Simple Parallel ------*/
	{
		start = clock();
		float * clusters = new float[D * CLUSTERS];
		memcpy(clusters, initialClusters, D * CLUSTERS * sizeof(float));
		uchar * membership = new uchar[nObj];
		memset(membership, 0, nObj);

		simpleParallel(obj, nObj, clusters, membership);

		check(actual, membership, clusters, nObj, nTest, testObj, testActual);
		printf("Simple Parallel Total execution time = %.2f\n", (clock() - start) * 1. / CLOCKS_PER_SEC);
	}

	/*------ End Simple Parallel -----*/



	/*------- Tiled Parallel ------*/
	{
		start = clock();
		float * clusters = new float[D * CLUSTERS];
		memcpy(clusters, initialClusters, D * CLUSTERS * sizeof(float));
		uchar * membership = new uchar[nObj];
		memset(membership, 0, nObj);

		tiledParallel(obj, nObj, clusters, membership);

		check(actual, membership, clusters, nObj, nTest, testObj, testActual);
		printf("Tiled Parallel Total execution time = %.2f\n", (clock() - start) * 1. / CLOCKS_PER_SEC);
	}
	/*------- End Tiled Parallel ------*/

	/*
	for (int i = 0; i < CLUSTERS; ++i) {
		string temp = to_string(i);
		temp += ".bmp";
		writeBMP(clusters + i * D, temp);
	}
	*/
	/*for (int i = 0; i < 1000; ++i) {
		cout << i <<  " is a member of " << (int)membership[i] << '\n';
	}*/

	
	system("PAUSE");
    return 0;
}

