// TEMPORARY: time HiGHS HFactor::build on a basis dumped from discopt's dual simplex.
#include "util/HFactor.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  HighsInt m; in >> m;
  std::vector<HighsInt> start{0}, index; std::vector<double> value;
  for (HighsInt j = 0; j < m; j++) {
    HighsInt k; in >> k;
    for (HighsInt t = 0; t < k; t++) { HighsInt r; double v; in >> r >> v; index.push_back(r); value.push_back(v); }
    start.push_back((HighsInt)index.size());
  }
  std::vector<double> times; HighsInt rank_def = -1;
  for (int rep = 0; rep < 50; rep++) {
    std::vector<HighsInt> basic(m);
    for (HighsInt j = 0; j < m; j++) basic[j] = j;  // every column of the dumped basis
    HFactor f;
    f.setup(m, m, start.data(), index.data(), value.data(), basic.data());
    auto t0 = std::chrono::steady_clock::now();
    rank_def = f.build();
    times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(times.begin(), times.end());
  printf("HFACTOR m=%d nnz=%zu reps=%zu median_ms=%.3f min_ms=%.3f rank_deficiency=%d\n",
         (int)m, index.size(), times.size(), times[25], times[0], (int)rank_def);
  return rank_def == 0 ? 0 : 1;
}
