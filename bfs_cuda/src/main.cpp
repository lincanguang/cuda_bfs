#include <iostream>
#include <chrono>

#include "graph/graph.h"
#include "cpu/bfs.h"
#include "gpu/simple/bfs_simple.cuh"
#include "gpu/quadratic/bfs_quadratic.cuh"
#include "gpu/scan/bfs_scan.cuh"

using namespace std;


void print(vector<int> &v);


class Checker {
	vector<int> expected_answer;
public:
	Checker(vector<int> exp_ans): expected_answer(exp_ans) {}

	pair<int, int> count_visited_vertices(const vector<int> &distance) {
		int depth = 0;
		int count = 0;
		for (int x : distance) {
			if (x < INT_MAX) {
				++count;
				if (x > depth) {
					depth = x;
				}
			}
		}
		return {count, depth};
	}

	void check(vector<int> answer) {
		assert(answer.size() == expected_answer.size());
		bool is_ok = true;
		int position_wrong = -1;
		int error_account = 0;
		for (int i = 0; i < answer.size(); ++i) {
			if (answer.at(i) != expected_answer.at(i)) {
				++error_account;
				is_ok = false;
				position_wrong = i;
				printf("error : Answer at %i equals %i but should be equal to %i\n", position_wrong, answer[position_wrong], expected_answer[position_wrong]);
				if (error_account > 10) {
					break;
				}
			}
		}
		if (is_ok) {
			pair<int, int> graph_output = count_visited_vertices(answer);
			int n_visited_vertices = graph_output.first;
			int depth = graph_output.second;
			printf("CHECKED SUCCESSFULY! Number of visited vertices: %i, depth: %i \n", n_visited_vertices, depth);
		}

			
			
	
	}
};


// Tests speed of a BFS algorithm
int main() {  // TODO: Add arguments to main program (type of graph, file path)
	Graph G(AdjacencyList, Directed);
	int startVertex;
	vector<int> distance;
	vector<bool> visited;

	startVertex = 1;

	// CPU
	distance = vector<int>(G.numVertices);
	
	visited = vector<bool>(G.numVertices);
	auto startTime = chrono::steady_clock::now();
	bfsCPU(startVertex, G, distance, visited);
	auto endTime = chrono::steady_clock::now();
	long duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for CPU implementation : %li micro second.\n", duration);
	Checker checker(distance);

	// mybfsCPU
	distance = vector<int>(G.numVertices);
	startTime = chrono::steady_clock::now();
	mybfsCPU(startVertex, G, distance, visited);
	endTime = chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for my CPU implementation : %li micro second.\n", duration);

	

	// Simple GPU

	distance = vector<int>(G.numVertices);
	startTime = chrono::steady_clock::now();
	bfsGPU(startVertex, G, distance, visited);
	endTime = std::chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for naive linear GPU implementation (with graph copying) : %li micro second.\n", duration);
	
	checker.check(distance);

	// my GPU, test the cost between cudaMalloc and cudaMallocHost
	/*
	distance = vector<int>(G.numVertices);
	startTime = chrono::steady_clock::now();
	mybfsGPU(startVertex, G, distance, visited);
	endTime = std::chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for my GPU implementation (with graph copying) : %li micro second.\n", duration);

	checker.check(distance);
	*/

	// Quadratic GPU

	distance = vector<int>(G.numVertices);
	startTime = chrono::steady_clock::now();
	bfsGPUQuadratic(startVertex, G, distance, visited);
	endTime = std::chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for quadratic GPU implementation (with graph copying) : %li micro second.\n", duration);

	checker.check(distance);

	// scan GPU

	distance = vector<int>(G.numVertices);
	startTime = chrono::steady_clock::now();
	bfsGPUScan(startVertex, G, distance, visited);
	endTime = std::chrono::steady_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	printf("Elapsed time for scan GPU implementation (with graph copying) : %li micro second.\n", duration);

	checker.check(distance);

//	print(distance);

	return 0;
}


void print(vector<int> &v) {
	cout << "{ ";
	for (int i = 0; i < v.size(); ++i) {
		cout << v[i];
		if (i < v.size() - 1)
			cout << ", ";
	}
	cout << " }" << endl;
}
