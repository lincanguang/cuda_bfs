# cuda_bfs
a implement of Breadth First Search on CUDA in Visual Studio 2017.

the framework of this projected is copied from https://github.com/kaletap/bfs-cuda-gpu

In this project, two new algorithm is added. 
For CPU bfs, the main cost is the push_back operation for std::vector. therefore, a array version queue is applied to reduce the cost of push back (see cuda_bfs/bfs_cuda/src/cpu/bfs.cpp -> mybfsCPU).

For GPU bfs, the tech detail is descripted in a Nvidia paper : https://research.nvidia.com/publication/scalable-gpu-graph-traversal (Merrill, Garland, Grimshaw). An impletation of scan bfs (Expande-Contract) has finished in folder cuda_bfs/bfs_cuda/src/gpu/scan. 

For a small graph (data/4elt.graph, with 15606 vertices), the scan algorithm with prefix sum passed the test.
when N_THREADS_PER_BLOCK == 256 (see src\gpu\param.h), the algorithm performed best.
But there is still something wrong with the bfs scan version when load a large graph(data/citationCiteseer.graph, with 268494 vertices).

## update in 2023/3/30
the bug memtioned above is about prefixed sum. warp_cull can drop many duplicated vertices, 
but it CAN'T guarentee all duplicated vertices are flitered. If there are three vertices v1,v2，v3
in the same warp, satisfied v3 = v2 = v1 + k * HASH_RANGE (k > 0), then the duplication between v2,v3 can't
be reduced because v2 doesn't saved in shared memory.

the performance for warp_cull showed in my(see cuda_bfs/bfs_cuda/src/gpu/simple -> mybfsGPU). 
