#include <cstdio>
#include <vector>
#include "util/HighsHash.h"
int main() {
  for (int n : {1, 2, 3, 7, 8, 63, 64, 65, 100}) {
    std::vector<HighsInt> v(n); for (int i = 0; i < n; ++i) v[i] = 3 * i + 1;
    std::vector<uint32_t> c(n); for (int i = 0; i < n; ++i) c[i] = HighsHashHelpers::double_hash_code(0.37 * (i + 1) - 1.0);
    printf("V %d %llu %llu\n", n, (unsigned long long)HighsHashHelpers::vector_hash(v.data(), n),
           (unsigned long long)HighsHashHelpers::vector_hash(c.data(), n));
  }
  for (double d : {1.0, 0.5, -0.25, 3.7, 1e-9, 0.0}) printf("D %a %u\n", d, HighsHashHelpers::double_hash_code(d));
}
