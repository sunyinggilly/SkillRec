#include "difficultyestimator.h"

double DifficultyEstimator::predict_easy(int s) {
	double ret = 0;
	for (int i = 0; i < active_edge[s].size(); i++) {
		Edge edge = active_edge[s][i];
		double pnow = (double)item_freq[edge.nxt] / item_freq[edge.from];
		ret = max(ret, pnow);
	}
	return log(ret);
}

void DifficultyEstimator::clear() {
	itemset_now.clear();
	itemset_now.push_back(0);
	for (int i = 0; i < 2000; i++) {
		active_edge[i].clear();
	}
	int sz = G[0].size();
	for (int i = 0; i < sz; i++) {
		Edge edge = G[0][i];
		active_edge[edge.s].push_back(edge);
	}
}

double DifficultyEstimator::predict_and_add(int s) {
	double ret = 0;
	for (int i = 0; i < active_edge[s].size(); i++) {
		Edge edge = active_edge[s][i];
		double pnow = (double)item_freq[edge.nxt] / item_freq[edge.from];
		ret = max(ret, pnow);
		for (int j = 0; j < G[edge.nxt].size(); j++) {
			Edge edge_new = G[edge.nxt][j];
			active_edge[edge_new.s].push_back(edge_new);
		}
	}
	active_edge[s].clear();
	return log(ret);
}
DifficultyEstimator::DifficultyEstimator(vector<vector<int>> item_sets, vector<int> item_freq, int n_samples) {
	int c = 0;
	map<vector<int>, int> itemset_id;
	itemset_id[vector<int>()] = c++;
	this->G.push_back(vector<Edge>());
	int n_set = item_sets.size();
	this->item_freq.push_back(n_samples);

	for (int i = 0; i < n_set; i++) {
		itemset_id[item_sets[i]] = c++;
		this->G.push_back(vector<Edge>());
		this->item_freq.push_back(item_freq[i]);
	}
	for (int i = 0; i < n_set; i++) {
		int sz = item_sets[i].size();
		int end_id = itemset_id[item_sets[i]];
		for (int j = 0; j < sz; j++) {
			vector<int> Vpar;
			for (int k = 0; k < sz; k++) if(j != k){
				Vpar.push_back(item_sets[i][k]);
			}
			int start_id = itemset_id[Vpar];
			this->G[start_id].push_back(Edge(item_sets[i][j], start_id, end_id));
		}
	}
}
Edge::Edge(int s, int from, int nxt) {
	this->s = s;
	this->nxt = nxt;
	this->from = from;
}
