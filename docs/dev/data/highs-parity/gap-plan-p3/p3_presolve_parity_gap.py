"""Presolve parity summary: rows whose reduced size differs from HiGHS by >2 rows or >5 cols,
and totals over the rows compared (prints the count, §6)."""
import json,sys
d=json.load(open(sys.argv[1] if len(sys.argv)>1 else __import__("os").path.dirname(__import__("os").path.abspath(__file__))+"/p3_presolve_parity.json"))
T=[0,0,0,0,0,0];n=0
for e in d:
  o=e['ours'];h=e['highs']
  if not(h and o): continue
  n+=1
  T[0]+=o[0];T[1]+=o[1];T[2]+=o[2];T[3]+=o[3];T[4]+=h[0];T[5]+=h[1]
  dr,dc=o[2]-h[0],o[3]-h[1]
  if dr>2 or dc>5: print(f"{e['name']:24s} ours {o[2]}x{o[3]}  highs {h[0]}x{h[1]}  gap {dr},{dc}")
print("rows compared",n,"orig",T[0],T[1],"ours",T[2],T[3],"highs",T[4],T[5])
