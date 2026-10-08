"""Table of final team return per (env, widths) over seeds; normalized per env-agent count by random policy."""
import glob, collections, torch, statistics as st

R = collections.defaultdict(list)
for p in glob.glob("runs/*_s*.pt"):
    if "smoke" in p: continue
    ck = torch.load(p, map_location="cpu")
    R[(ck["args"]["env"], ck["args"]["widths"])].append(ck["final"])
for (e, w), v in sorted(R.items()):
    n = len(w.split(","))
    print(f"{e:18s} {w:10s} n={len(v)}  J={st.mean(v):8.3f} ± {st.pstdev(v):.3f}   J/agent={st.mean(v)/n:7.3f}   seeds={[round(x,2) for x in v]}")
