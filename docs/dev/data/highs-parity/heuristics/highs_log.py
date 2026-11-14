import sys, highspy
path, log = sys.argv[1], sys.argv[2]
h = highspy.Highs()
for k, v in (("output_flag", True), ("log_to_console", False), ("log_file", log),
             ("threads", 1), ("time_limit", 20.0), ("mip_rel_gap", 1e-4)):
    h.setOptionValue(k, v)
h.readModel(path)
h.run()
