import os, sys, pickle
os.environ["DISCOPT_PROFILE"] = "1"
import os as _os, pathlib as _pl  # portable: repo root from this file, or $DISCOPT_ROOT
W = _os.environ.get("DISCOPT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
CAP = _os.environ.get("G32_CAPTURES") or str(_pl.Path(__file__).resolve().parent / "captures")
import discopt; assert discopt.__file__.startswith(W + "/python")
import discopt._rust as R
d = pickle.load(open(CAP + "/loa__med_term_purchasing.pkl", "rb")); d.pop("had_debug_hook", None); d["initial_incumbent"] = None
os.environ["DISCOPT_RUST_MILP_COEF_TIGHTEN"] = sys.argv[1]; os.environ["DISCOPT_RUST_MILP_TABLEAU_CMIR"] = sys.argv[5]
k = dict(d); k.update(root_cuts=int(sys.argv[2]), cut_rounds=int(sys.argv[3]), cut_select=sys.argv[4] == "1", max_nodes=2, time_limit_s=300.0)
R.profile_reset_py(); r = R.solve_milp_csc_py(**k); pc = R.profile_counters_py()
print("root", r[3]); print({kk: v for kk, v in pc.items() if ("Cut" in kk or "Root" in kk or "Cmir" in kk or "Subst" in kk or "Dedup" in kk) and v})
