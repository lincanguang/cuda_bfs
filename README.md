# cuda_bfs
a implement of Breadth First Search on CUDA in Visual Studio 2017.

the framework of this projected is copied from https://github.com/kaletap/bfs-cuda-gpu

In this project, two new algorithm is added. 
For CPU bfs, the main cost is the push_back operation for std::vector. therefore, a array version queue is applied to reduce the cost of push back (see cuda_bfs/bfs_cuda/src/cpu/bfs.cpp -> mybfsCPU).

For GPU bfs, the tech detail is descripted in a Nvidia paper : https://research.nvidia.com/publication/scalable-gpu-graph-traversal (Merrill, Garland, Grimshaw). An impletation of scan bfs (Expande-Contract) has finished in folder cuda_bfs/bfs_cuda/src/gpu/scan. 

there is still something wrong with the bfs scan version when load a large graph.
