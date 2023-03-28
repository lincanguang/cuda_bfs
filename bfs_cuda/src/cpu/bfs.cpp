#include "bfs.h"

using namespace std;


void bfsCPU(int start, Graph &G, std::vector<int> &distance, std::vector<bool> &visited) {
	fill(distance.begin(), distance.end(), INT_MAX);
	distance[start] = 0;
	queue<int> to_visit;
	to_visit.push(start);

	while (!to_visit.empty()) {
		int current = to_visit.front();
		to_visit.pop();
		for (int i = G.edgesOffset[current]; i < G.edgesOffset[current] + G.edgesSize[current]; ++i) {
			int v = G.adjacencyList[i];
			if (distance[v] == INT_MAX) {
				distance[v] = distance[current] + 1;
				to_visit.push(v);
			}
		}
	}
}


void mybfsCPU(int start, Graph &G, vector<int> &distance, vector<bool> &visited) {
	fill(distance.begin(), distance.end(), INT_MAX);
	int n = G.numVertices;
	vector<int> Q(n+1);

	unsigned int front = 0, rear = 0;

	//push start
	Q[rear++] = start;
	int current = 0;
	while ( rear - front > 0 && rear < n) {
		current = Q[front++];
		for (int i = G.edgesOffset[current]; i < G.edgesOffset[current] + G.edgesSize[current]; ++i) {
			int v = G.adjacencyList[i];
			if (distance[v] == INT_MAX) {
				distance[v] = distance[current] + 1;
				Q[rear++] = v;
			}
		}
	}

}