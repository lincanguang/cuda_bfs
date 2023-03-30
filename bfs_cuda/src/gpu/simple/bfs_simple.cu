#include "bfs_simple.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)



__global__
void initializeDeviceArray(int n, int *d_arr, int value, int start_index) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == start_index) {
		d_arr[start_index] = 0;
	}
	else if (tid < n) {
		d_arr[tid] = value;
	}
}


__global__
void printDeviceArray(int *d_arr, int n) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		printf("d_arr[%i] = %i \n", tid, d_arr[tid]);
	}
}

__device__ int warp_cull_(volatile int scratch[WARPS][HASH_RANGE], const int v)
{
	//unsigned int active = __ballot_sync(FULL_MASK, v >= 0);
	//if( v == -1) return v; 
	const int hash = v & (HASH_RANGE - 1);
	const int warp_id = threadIdx.x / WARP_SIZE;

	if (v >= 0)
		scratch[warp_id][hash] = v;
	__syncwarp();
	const int retrieved = v >= 0 ? scratch[warp_id][hash] : v;
	__syncwarp();
	unsigned int active = __ballot_sync(0xffffffff, retrieved == v);
	if (retrieved == v)
	{
		// Vie to be the only thread in warp inspecting vertex v.
		scratch[warp_id][hash] = threadIdx.x;
		__syncwarp(active);
		// Some other thread has this vertex
		if (scratch[warp_id][hash] != threadIdx.x)
			return -1;
	}
	return v;
}

/*
 * Given a graph and a current queue computes next vertices (vertex frontiers) to traverse.
 */
__global__
void computeNextQueue(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
		int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id

	if (tid < queueSize) {  // visit all vertexes in a queue in parallel
		int current = currentQueue[tid];
		for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i) {
			int v = adjacencyList[i];
			if (distance[v] == INT_MAX) {
				distance[v] = level + 1;
				int position = atomicAdd(nextQueueSize, 1);
				//if (tid < 10) {
					//printf("t, pos:(%d,%d)   ", tid, position);
				//}
				nextQueue[position] = v;
			}
		}
	}
}

__global__
void myComputeNextQueue(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
	int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id

	volatile __shared__ int scratch[WARPS][HASH_RANGE];
	

	if (tid < queueSize) {  // visit all vertexes in a queue in parallel
		int current = currentQueue[tid];
		current = warp_cull_(scratch, current);
		if (current >= 0) {
			for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i) {
				int v = adjacencyList[i];
				if (distance[v] == INT_MAX) {
					distance[v] = level + 1;
					int position = atomicAdd(nextQueueSize, 1);
					//if (tid < 10) {
						//printf("t, pos:(%d,%d)   ", tid, position);
					//}
					nextQueue[position] = v;
				}
			}
		}
	}
}



void bfsGPU(int start, Graph &G, vector<int> &distance, vector<bool> &visited) {
	const int n_blocks = (G.numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
	// Initialization of GPU variables
	int *d_adjacencyList;
	int *d_edgesOffset;
	int *d_edgesSize;
	int *d_distance; // output

	int *d_firstQueue;
	int *d_secondQueue;
	int *d_nextQueueSize;
	
	// Initialization of CPU variables
	int currentQueueSize = 1;
	const int NEXT_QUEUE_SIZE = 0;
	int level = 0;

	// Allocation on device
	const int size = G.numVertices * sizeof(int);
	const int adjacencySize = G.adjacencyList.size() * sizeof(int);
	cudaMalloc((void **)&d_adjacencyList, adjacencySize);
	cudaMalloc((void **)&d_edgesOffset, size);
	cudaMalloc((void **)&d_edgesSize, size);
	cudaMalloc((void **)&d_firstQueue, size);
	cudaMalloc((void **)&d_secondQueue, size);
	cudaMalloc((void **)&d_distance, size);
	cudaMalloc((void **)&d_nextQueueSize, sizeof(int));


	// Copy inputs to device
	cudaMemcpy(d_adjacencyList, &G.adjacencyList[0], adjacencySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesOffset, &G.edgesOffset[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesSize, &G.edgesSize[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);
//	initializeDeviceArray<<<n_blocks, N_THREADS_PER_BLOCK>>> (G.numVertices, d_distance, INT_MAX, start); // FOR SOME REASON USING THIS KERNEL DOESNT WORK
//	cudaDeviceSynchronize();

	auto startTime = chrono::steady_clock::now();
	distance = vector<int> (G.numVertices, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

	while (currentQueueSize > 0) {
		printf("qsize, level := %d, %d \n", currentQueueSize, level);
		int *d_currentQueue;
		int *d_nextQueue;
		if (level % 2 == 0) {
			d_currentQueue = d_firstQueue;
			d_nextQueue = d_secondQueue;
		}
		else {
			d_currentQueue = d_secondQueue;
			d_nextQueue = d_firstQueue;
		}
		computeNextQueue<<<n_blocks, N_THREADS_PER_BLOCK>>> (d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
				currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
		cudaDeviceSynchronize();
		++level;
		cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	}

	// Copying output back to host
	cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);
	auto endTime = std::chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for naive linear GPU implementation (without copying graph) : %li micro second.\n", duration);

	// Cleanup
	cudaFree(d_adjacencyList);
	cudaFree(d_edgesOffset);
	cudaFree(d_edgesSize);
	cudaFree(d_firstQueue);
	cudaFree(d_secondQueue);
	cudaFree(d_distance);
	cudaFree(d_nextQueueSize);
}


void mybfsGPU(int start, Graph &G, vector<int> &distance, vector<bool> &visited) {
	const int n_blocks = (G.numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
	// Initialization of GPU variables
	int *d_adjacencyList;
	int *d_edgesOffset;
	int *d_edgesSize;
	int *d_distance; // output

	int *d_firstQueue;
	int *d_secondQueue;
	int *d_nextQueueSize;

	// Initialization of CPU variables
	int currentQueueSize = 1;
	const int NEXT_QUEUE_SIZE = 0;
	int level = 0;

	// Allocation on device
	const int size = G.numVertices * sizeof(int);
	const int adjacencySize = G.adjacencyList.size() * sizeof(int);
	cudaMalloc((void **)&d_adjacencyList, adjacencySize);
	cudaMalloc((void **)&d_edgesOffset, size);
	cudaMalloc((void **)&d_edgesSize, size);
	cudaMalloc((void **)&d_firstQueue, size);
	cudaMalloc((void **)&d_secondQueue, size);
	cudaMalloc((void **)&d_distance, size);
	cudaMalloc((void **)&d_nextQueueSize, sizeof(int));


	// Copy inputs to device
	cudaMemcpy(d_adjacencyList, &G.adjacencyList[0], adjacencySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesOffset, &G.edgesOffset[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesSize, &G.edgesSize[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);
	//	initializeDeviceArray<<<n_blocks, N_THREADS_PER_BLOCK>>> (G.numVertices, d_distance, INT_MAX, start); // FOR SOME REASON USING THIS KERNEL DOESNT WORK
	//	cudaDeviceSynchronize();

	auto startTime = chrono::steady_clock::now();
	distance = vector<int>(G.numVertices, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

	while (currentQueueSize > 0) {
		//printf("qsize, level := %d, %d \n", currentQueueSize, level);
		int *d_currentQueue;
		int *d_nextQueue;
		if (level % 2 == 0) {
			d_currentQueue = d_firstQueue;
			d_nextQueue = d_secondQueue;
		}
		else {
			d_currentQueue = d_secondQueue;
			d_nextQueue = d_firstQueue;
		}
		myComputeNextQueue << <n_blocks, N_THREADS_PER_BLOCK >> > (d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
			currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
		cudaDeviceSynchronize();
		++level;
		cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	}

	// Copying output back to host
	cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);
	auto endTime = std::chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for my linear GPU implementation (without copying graph) : %li micro second.\n", duration);

	// Cleanup
	cudaFree(d_adjacencyList);
	cudaFree(d_edgesOffset);
	cudaFree(d_edgesSize);
	cudaFree(d_firstQueue);
	cudaFree(d_secondQueue);
	cudaFree(d_distance);
	cudaFree(d_nextQueueSize);
}