"""Run HiGHS on MIPLIB rows, 20 s, and print the log lines that record each improving solution."""
import sys, highspy, os
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
pop = {m["name"]: m for m in R.population()}
n = 0
for name in sys.argv[1].split(","):
    h = highspy.Highs()
    log = f"{sys.argv[2]}/{name}.highs.log"
    h.setOptionValue("log_file", log); h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False); h.setOptionValue("time_limit", 20.0)
    h.setOptionValue("threads", 1)
    h.readModel("/home/user/discopt/" + pop[name]["path"]); h.run(); n += 1
    print("==", name, h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value, flush=True)
print("solved", n)
