#include <map>
#include <stack>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
using namespace std;
const int maxnode = 1e6;
class Edge {
public:
	int s, from, nxt;
	Edge(int s, int from, int nxt);
};
class DifficultyEstimator {
public:
	vector<int> item_freq;
	vector<vector<Edge>> G;
	vector<int> itemset_now;
	vector<Edge> active_edge[2000];
	void clear();
	DifficultyEstimator(vector<vector<int>> item_sets, vector<int> item_freq, int n_samples);
	double predict_easy(int s);
	double predict_and_add(int s);
};
