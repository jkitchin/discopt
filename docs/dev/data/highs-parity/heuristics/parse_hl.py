import sys, re, collections, os
SRC = re.compile(r"^ ([A-Za-z]) +\d+ +\d+ +\d+ +[\d.]+% +\S+ +(\S+) .*?(\d+\.\d+)s\s*$")
first = collections.Counter(); anyc = collections.Counter(); rows = []
for f in sys.argv[1:]:
    name = os.path.basename(f)[3:-4]
    sols = []
    for line in open(f):
        m = SRC.match(line)
        if m and m.group(2) != "inf":
            sols.append((m.group(1), float(m.group(3))))
    rows.append((name, sols[0] if sols else None, "".join(s for s, _ in sols)))
    if sols: first[sols[0][0]] += 1
    for s in set(x for x, _ in sols): anyc[s] += 1
for r in sorted(rows, key=lambda r: (r[1] or ("~", 99))[1]): print(f"{r[0]:28} first={r[1]}  all={r[2]}")
print("first-solution source:", dict(first)); print("rows using source:", dict(anyc))
