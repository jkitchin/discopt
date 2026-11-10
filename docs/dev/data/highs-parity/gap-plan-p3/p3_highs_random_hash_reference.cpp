#include <cstdio>
#include <utility>
#include "util/HighsRandom.h"
#include "util/HighsCDouble.h"
int main() {
  for (int a : {0, 1, 7, 12345, -3}) for (int b : {0, 42, 2147483647, -1})
    printf("H %d %d %llu\n", a, b, (unsigned long long)HighsHashHelpers::hash(std::make_pair(a, b)));
  for (unsigned seed : {0u, 1u, 12345u, 4000000000u}) {
    HighsRandom r(seed);
    printf("R %u %llu", seed, (unsigned long long)r.debugState());
    for (int k = 0; k < 3; ++k) printf(" %d", r.integer());
    printf(" %d %d %.17g %.17g %d\n", r.integer(10), r.integer(1000003), r.fraction(), r.closedFraction(), (int)r.bit());
  }
}
