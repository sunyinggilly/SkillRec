#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <cmath>
#include <cstring>
using namespace std;

class State {
public:
	int samp_id;
	double salary;
	State(int sampid = 0, double salary = 0);
	friend bool operator < (State a, State b);
	friend bool operator == (State a, State b);
};
class JobMatcher {
    private:
		int w, n_samples, n_top;
        double th;
		vector<vector<int> > skill_list;
		vector<double> salary;
		vector<int> skill_sample_list[2000];
		int p[600000] = { 0 };
		priority_queue<State> Qs;
    public:
		JobMatcher(int w, double th, int n_top, vector<vector<int> > skill_list, vector<double> salary);
		void reset();
		double discount(int n_cur, int n_all);
		void add(int s);
		double predict_salary(int s);
		double top_average_of_heap(priority_queue<State> Qs_tmp);
		double top_average();
};
