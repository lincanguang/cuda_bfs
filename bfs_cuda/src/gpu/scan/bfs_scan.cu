#include "bfs_scan.cuh"

using namespace std;

constexpr unsigned int FULL_MASK = 0xffffffff;
struct prescan_result
{
	int offset, total;
};

int div_up(int dividend, int divisor);
// bfs fronter scan,  Scalable GPU Graph Traversal

/* ===========================    __device__ function ============================  */

__device__ prescan_result block_prefix_sum(const int val)
{
	// Heavily inspired/copied from sample "shfl_scan" provided by NVIDIA.
	// Block-wide prefix sum using shfl intrinsic.
	volatile __shared__ int sums[WARPS];
	int value = val;

	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;

	// Warp-wide prefix sums.
#pragma unroll
	for (int i = 1; i <= WARP_SIZE; i <<= 1)
	{
		const int n = __shfl_up_sync(FULL_MASK, value, i, WARP_SIZE);
		if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array.
	if (lane_id == WARP_SIZE - 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	// Prefix sum of warp sums.
	if (warp_id == 0 && lane_id < WARPS)
	{
		int warp_sum = sums[lane_id];
		const unsigned int mask = (1 << (WARPS)) - 1;
#pragma unroll
		for (int i = 1; i <= WARPS; i <<= 1)
		{
			const int n = __shfl_up_sync(mask, warp_sum, i, WARPS);
			if (lane_id >= i)
				warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// Add total sum of previous warps to current element.
	if (warp_id > 0)
	{
		const int block_sum = sums[warp_id - 1];
		value += block_sum;
	}

	prescan_result result;
	// Subtract value given by thread to get exclusive prefix sum.
	result.offset = value - val;
	// Get total sum.
	result.total = sums[WARPS - 1];
	return result;
}

__device__ int warp_cull(volatile int scratch[WARPS][HASH_RANGE], const int v)
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
	unsigned int active = __ballot_sync(FULL_MASK, retrieved == v);
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

__device__ void block_gather(
	int* adjacencyList, int* distance,
	int level, int * out_queue,
	int* out_queue_count, int r, int r_end)
{
	volatile __shared__ int comm[3];
	while (__syncthreads_or(r < r_end))
	{
		// Vie for control of block.
		if (r < r_end)
			comm[0] = threadIdx.x;
		__syncthreads();
		if (comm[0] == threadIdx.x)
		{
			// If won, share your range to the entire block.
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
		}
		__syncthreads();
		int r_gather = comm[1] + threadIdx.x;
		const int r_gather_end = comm[2];
		const int total = comm[2] - comm[1];
		int block_progress = 0;
		while ((total - block_progress) > 0)
		{
			int neighbor = -1;
			bool is_valid = false;
			if (r_gather < r_gather_end)
			{
				neighbor = adjacencyList[r_gather];
				// Look up status of current neighbor.
				volatile bool is_valid = (distance[neighbor] == INT_MAX);
				if (is_valid)
				{
					// Update label.
					distance[neighbor] = level + 1;
				}
			}
			// Obtain offset in queue by computing prefix sum
			const prescan_result prescan = block_prefix_sum(is_valid ? 1 : 0);
			volatile __shared__ int base_offset[1];

			// Obtain base enqueue offset and share it to whole block.
			if (threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count, prescan.total);
			__syncthreads();
			// Write vertex to the out queue.
			if (is_valid)
				out_queue[base_offset[0] + prescan.offset] = neighbor;

			r_gather += N_THREADS_PER_BLOCK;
			block_progress += N_THREADS_PER_BLOCK;
			__syncthreads();
		}
	}
}

__device__ void fine_gather(int* adjacencyList, int* distance,
	int level, int * out_queue, int* out_queue_count, int r, int r_end)
{
	prescan_result rank = block_prefix_sum(r_end - r);
	__shared__ int comm[N_THREADS_PER_BLOCK];
	int cta_progress = 0;

	while ((rank.total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists.
		while ((rank.offset < cta_progress + N_THREADS_PER_BLOCK) && (r < r_end))
		{
			comm[rank.offset - cta_progress] = r;
			rank.offset++;
			r++;
		}
		__syncthreads();
		int neighbor;
		bool is_valid = false;
		if (threadIdx.x < (rank.total - cta_progress))
		{
			neighbor = adjacencyList[comm[threadIdx.x]];
			// Look up status
			is_valid = (distance[neighbor] == INT_MAX);;
			if (is_valid)
			{
				// Update label
				distance[neighbor] = level + 1;
			}
		}
		__syncthreads();
		// Obtain offset in queue by computing prefix sum.
		const prescan_result prescan = block_prefix_sum(is_valid ? 1 : 0);
		volatile __shared__ int base_offset[1];
		// Obtain base enqueue offset
		if (threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count, prescan.total);
		}
		__syncthreads();
		const int queue_index = base_offset[0] + prescan.offset;
		// Write to queue
		if (is_valid)
		{
			out_queue[queue_index] = neighbor;
		}

		cta_progress += N_THREADS_PER_BLOCK;
		__syncthreads();
	}
}

/* ===========================    __global__ function ============================  */

__global__ void kernal_bfs_scan(
	int *d_adjacencyList,
	int *d_edgesSize,
	int *d_edgesOffset,
	int *d_distance,
	int *d_currentQueue,
	int *d_nextQueue,
	int *d_nextQueueSize,
	int num_vertices, int qsize, int level)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	do {
		int v = tid < qsize ? d_currentQueue[tid] : -1;
		// do local warp-culling
		volatile __shared__ int scratch[WARPS][HASH_RANGE];
		v = warp_cull(scratch, v);
		
		const int r = v < 0 ? 0 : d_edgesOffset[v];
		const int r_end = v < 0 ? 0 : d_edgesOffset[v]+ d_edgesSize[v];
		const bool big_list = (r_end - r) >= N_THREADS_PER_BLOCK;

		block_gather(d_adjacencyList, d_distance, level, d_nextQueue, d_nextQueueSize, r, big_list ? r_end : r);
		fine_gather(d_adjacencyList, d_distance, level, d_nextQueue, d_nextQueueSize, r, big_list ? r : r_end);

		tid += gridDim.x*blockDim.x;
	} while (__syncthreads_or(tid < qsize));
}




// Assumes that distance is a vector of all INT_MAX (except at start position)
void bfsGPUScan(int start, Graph &G, vector<int> &distance, vector<bool> &visited) {
	// Initialization of GPU variables
	int *d_adjacencyList;
	int *d_edgesOffset;
	int *d_edgesSize;
	int *d_distance; // output

	int *d_queue; 
	int *d_nextQueue;
	int *d_qsize;

	// Initialization of CPU variables
	int qsize = 1;
	int level = 0;

	// Allocation on device
	const int size = G.numVertices * sizeof(int);
	const int adjacencySize = G.adjacencyList.size() * sizeof(int);
	cudaMalloc((void **)&d_adjacencyList, adjacencySize);
	cudaMalloc((void **)&d_edgesOffset, size);
	cudaMalloc((void **)&d_edgesSize, size);
	cudaMalloc((void **)&d_distance, size);

	cudaMalloc((void **)&d_queue, size);
	cudaMalloc((void **)&d_nextQueue, size);

	cudaMalloc((void **)&d_qsize, sizeof(int));
	
	distance = vector<int>(G.numVertices, INT_MAX);
	distance[start] = 0;
	

	// Copy inputs to device
	cudaMemcpy(d_adjacencyList, &G.adjacencyList[0], adjacencySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesOffset, &G.edgesOffset[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesSize, &G.edgesSize[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_queue, &start, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_qsize, &qsize, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

	auto startTime = chrono::steady_clock::now();
	
	while (qsize) {
		printf("qsize, level := %d, %d \n", qsize, level);
		cudaMemset(d_qsize, 0, sizeof(int));
		const int num_of_blocks = div_up(qsize,
			QUEUE_RATIO_EXPAND_CONTRACT * N_THREADS_PER_BLOCK);

		kernal_bfs_scan << <num_of_blocks, N_THREADS_PER_BLOCK >> > (
			d_adjacencyList, d_edgesSize, d_edgesOffset, d_distance,
			d_queue, d_nextQueue, d_qsize,
			G.numVertices, qsize, level);
		cudaMemcpy(&qsize, d_qsize, sizeof(int), cudaMemcpyDeviceToHost);
		std::swap(d_queue, d_nextQueue);
		++level;
	}

	// Copying output back to host
	cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);
	auto endTime = std::chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for scan GPU implementation (without copying graph) : %li micro seconds.\n", duration);

	// Cleanup
	cudaFree(d_adjacencyList);
	cudaFree(d_edgesOffset);
	cudaFree(d_edgesSize);
	cudaFree(d_distance);
	cudaFree(d_queue);
	cudaFree(d_nextQueue);
}


int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}