%module DifficultyEstimatorGLinux
%include "std_string.i"
%include "std_vector.i"
%{
#include "difficultyestimator.h"
%}

namespace std {
  %template(StringVector) vector<string>;
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(DoubleVector) vector<double>;
}

%include "difficultyestimator.h"
