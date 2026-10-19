#!/usr/bin/env bash
# Open solvers for the #1775 GDP baseline (G0.6, spec §1.2 "profile O"):
# HiGHS (MIP), Ipopt (NLP), SCIP (MINLP / global), all reachable from Pyomo.
#
# GDPlib's own benchmark.py has GAMS profiles only, so the open profile is run by
# discopt_benchmarks/scripts/gdp_baseline.py, and this script provides its solvers
# on an Ubuntu 24.04 machine without conda:
#
#   * HiGHS   -- pip highspy (Pyomo: appsi_highs)
#   * CBC     -- apt coinor-cbc (Pyomo: cbc). GDPopt's MIP role: Pyomo's HiGHS
#                interfaces raise on the dual suffix GDPopt attaches (see
#                gdp_baseline.py).
#   * SCIP    -- pip pyscipopt, which bundles libscip (Pyomo: scip_direct), and the
#                SCIP executable built from source with Ipopt (Pyomo: scip), which
#                GDPopt-GLOA needs as its global NLP solver
#   * Ipopt   -- apt coinor-libipopt-dev has the library but no executable, so the
#                AMPL driver is compiled from the matching Ipopt release against it
#                and against ASL built from github.com/ampl/asl (Pyomo: ipopt). The
#                same ASL serves PyNumero, so Pyomo's cyipopt works too.
#
# Needs root (apt, /opt, /usr/local/bin). Idempotent. Verified 2026-10-19.
set -euo pipefail

WORK="${WORK:-$(mktemp -d)}"
echo "building in $WORK"

apt-get update -qq
apt-get install -y -qq coinor-libipopt-dev coinor-cbc liblapack-dev libblas-dev gfortran \
  pkg-config cmake g++ git

python -m pip install -q highspy pyscipopt cyipopt

# ASL (AMPL Solver Library), shared, into /opt/asl
if [ ! -f /opt/asl/lib/libasl.so ]; then
  git clone -q --depth 1 https://github.com/ampl/asl.git "$WORK/asl"
  cmake -S "$WORK/asl" -B "$WORK/asl-build" -DCMAKE_INSTALL_PREFIX=/opt/asl \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_MT_LIBS=OFF \
    -DBUILD_EXAMPLES=OFF > /dev/null
  make -C "$WORK/asl-build" -j"$(nproc)" install > /dev/null
fi
echo /opt/asl/lib > /etc/ld.so.conf.d/asl.conf
ldconfig

# The Ipopt AMPL driver, from the release matching the apt library.
IPOPT_VERSION="$(pkg-config --modversion ipopt)"
if ! command -v ipopt > /dev/null; then
  git clone -q --depth 1 --branch "releases/$IPOPT_VERSION" --filter=blob:none --sparse \
    https://github.com/coin-or/Ipopt.git "$WORK/ipopt"
  git -C "$WORK/ipopt" sparse-checkout set Ipopt/src
  SRC="$WORK/ipopt/Ipopt/src"
  # Ubuntu's -dev package omits some internal headers; take them from the same release.
  INC=$(find "$SRC" -name '*.hpp' -printf '-I%h\n' | sort -u | tr '\n' ' ')
  g++ -O2 -DHAVE_CSTDDEF -DHAVE_CSTDIO -DHAVE_CSTRING -DHAVE_CSTDLIB \
    -I"$SRC/Apps/AmplSolver" -I/usr/include/coin $INC -I/opt/asl/include/asl \
    "$SRC/Apps/AmplSolver/ampl_ipopt.cpp" "$SRC/Apps/AmplSolver/AmplTNLP.cpp" \
    -o /usr/local/bin/ipopt -L/opt/asl/lib -lasl $(pkg-config --libs ipopt) -ldl
fi

# The SCIP executable (SoPlex 8.0.3 + SCIP 10, branch v10-minor), for Pyomo's ``scip``.
# Without Ipopt: SCIP needs Ipopt >= 3.12 and Ubuntu ships 3.11.9. SCIP still solves
# nonconvex NLP/MINLP globally (spatial branch-and-bound on LP relaxations); it only
# loses its local-NLP heuristics. (Tags like v1009 are SCIP 1.0.0.9, not 10.0.9.)
if [ ! -x /opt/scip/bin/scip ]; then
  git clone -q --depth 1 --branch v8.0.3 https://github.com/scipopt/soplex.git "$WORK/soplex"
  git clone -q --depth 1 --branch v10-minor https://github.com/scipopt/scip.git "$WORK/scip"
  cmake -S "$WORK/soplex" -B "$WORK/soplex/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/scip -DGMP=OFF -DBOOST=OFF -DMPFR=OFF > /dev/null
  make -C "$WORK/soplex/build" -j"$(nproc)" install > /dev/null
  cmake -S "$WORK/scip" -B "$WORK/scip/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/scip -DSOPLEX_DIR=/opt/scip -DIPOPT=OFF \
    -DPAPILO=OFF -DZIMPL=OFF -DGMP=OFF -DREADLINE=OFF -DAMPL=ON -DBOOST=OFF -DTPI=none \
    -DAUTOBUILD=OFF > /dev/null
  make -C "$WORK/scip/build" -j"$(nproc)" scip > /dev/null
  make -C "$WORK/scip/build" install > /dev/null
fi
ln -sf /opt/scip/bin/scip /usr/local/bin/scip

# MC++ (Pyomo's mcpp plugin), which GDPopt-GLOA uses for its McCormick cuts. Pyomo's
# own build goes through setuptools and fails on Debian's Python ("install_layout");
# the plugin is one C++ file loaded with ctypes, so compile it directly.
mkdir -p "$HOME/.pyomo/lib"
if [ ! -f "$HOME/.pyomo/lib/mcppInterface.so" ]; then
  git clone -q --depth 1 https://github.com/omega-icl/mcpp.git "$WORK/mcpp"
  MCPP_SRC="$(python -c 'import pyomo.contrib.mcpp, os; print(os.path.dirname(pyomo.contrib.mcpp.__file__))')"
  g++ -shared -fPIC -O2 -std=c++17 -I"$WORK/mcpp/src/mc" -I"$WORK/mcpp/src/3rdparty/fadbad++" \
    "$MCPP_SRC/mcppInterface.cpp" -o "$HOME/.pyomo/lib/mcppInterface.so"
fi

# PyNumero's ASL interface, for Pyomo's cyipopt.
python - <<'EOF'
from pyomo.contrib.pynumero.build import build_pynumero
build_pynumero(user_args=["-DAMPLASL_DIR=/opt/asl", "-DBUILD_AMPLASL_IF_NEEDED=OFF",
                          "-DBUILD_AMPLASL=OFF"], parallel=4)
EOF

python - <<'EOF'
import pyomo.environ as pyo
for s in ("appsi_highs", "cbc", "scip_direct", "scip", "ipopt", "cyipopt"):
    ok = pyo.SolverFactory(s).available(exception_flag=False)
    print(f"{s:12s} {'ok' if ok else 'MISSING'}")
    assert ok, s
EOF
python -c "from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available as a; assert a(); print('mcpp         ok')"
ipopt -v | head -1
scip --version | head -1
