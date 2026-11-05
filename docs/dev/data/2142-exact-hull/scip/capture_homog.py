"""#2142: capture the kernel spec of cstr under the division-free (homog) hull, for SCIP.
Installs exact_hull_proto's 'homog' arm, then runs 2142-scip-expert/py/capture.py."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import exact_hull_proto as P
P.install("homog")
os.environ["GDP"] = "hull"
sys.argv = ["capture.py", "--capture", sys.argv[1]]
runpy.run_path(os.path.join(HERE, "..", "..", "2142-scip-expert", "py", "capture.py"), run_name="__main__")
