#ifndef PARAM_H_
#define PARAM_H_

#define DEBUG(x)

//BIT_THREADS_PER_BLOCK <= 10 because N_THREADS_PER_BLOCK <= 1024
constexpr size_t BIT_THREADS_PER_BLOCK = 5;
constexpr size_t WARP_SIZE = 32;

constexpr size_t N_THREADS_PER_BLOCK = (1 << BIT_THREADS_PER_BLOCK);

constexpr size_t WARPS = N_THREADS_PER_BLOCK / WARP_SIZE;
constexpr size_t HASH_RANGE = N_THREADS_PER_BLOCK;

// By changing these, you can tune how many vertices/edges from the queue will
// be processed by one thread.
constexpr int QUEUE_RATIO_EXPAND_CONTRACT = 1;

#endif // PARAM_H_