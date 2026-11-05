"""#2142 Part B entry check (PREREG.md): closure of the kernel-vs-SCIP box gap by the reverse step.
python -u b4_compare.py BOXES.npz K.npz KR.npz S.npz"""

import sys
import numpy as np

d, K, KR, S = (dict(np.load(f)) for f in sys.argv[1:5])
assert str(K["reverse"]) in ("", "0") and str(KR["reverse"]) == "1", (K["reverse"], KR["reverse"])
n = int(d["n_cols"])
n_orig = int(d["n_orig"])
glo, ghi = d["global_lo"].astype(float), d["global_hi"].astype(float)
rootw = np.where((glo > -1e19) & (ghi < 1e19), ghi - glo, np.nan)


def fin(lo, hi):
    lo = np.where(lo <= -1e19, -np.inf, lo)
    hi = np.where(hi >= 1e19, np.inf, hi)
    return lo, hi


n_cmp = 0
for tag in ("rinf", "r15"):
    tot = {"K": 0.0, "KR": 0.0}
    cnt = dict(
        boxes=0,
        scip_empty=0,
        K_empty=0,
        KR_empty=0,
        K_empty_where_scip_empty=0,
        KR_empty_where_scip_empty=0,
        boxes_KR_tighter_than_K=0,
        boxes_K_tighter_than_KR=0,
    )
    per_col = np.zeros(n)
    for b in range(len(d["rec_id"])):
        cnt["boxes"] += 1
        s_live = bool(S["live"][b])
        exK = np.zeros(n)
        cnt["scip_empty"] += int(not s_live)
        for A, src in (("K", K), ("KR", KR)):
            live = bool(src[f"{tag}_live"][b])
            cnt[f"{A}_empty"] += int(not live)
            if not s_live:
                cnt[f"{A}_empty_where_scip_empty"] += int(not live)
                tot[A] += 0.0 if not live else float(n)
                continue
            if not live:
                if A == "K":
                    exK = np.zeros(n)
                continue  # A proved empty where SCIP did not: no excess
            lo, hi = fin(src[f"{tag}_lo"][b], src[f"{tag}_hi"][b])
            slo, shi = S["lo"][b], S["hi"][b]
            wa, ws = hi - lo, shi - slo
            ok = np.isfinite(ws) & np.isfinite(rootw) & (rootw > 0)
            ex = np.zeros(n)
            ex[ok] = np.clip(
                np.where(np.isfinite(wa[ok]), (wa[ok] - ws[ok]) / rootw[ok], 1.0), 0.0, 1.0
            )
            tot[A] += float(ex.sum())
            n_cmp += int(ok.sum())
            if A == "K":
                exK = ex
            else:
                per_col += np.maximum(exK - ex, 0.0)
            if A == "KR" and bool(K[f"{tag}_live"][b]):
                cnt["boxes_KR_tighter_than_K"] += int(ex.sum() < exK.sum() - 1e-12)
                cnt["boxes_K_tighter_than_KR"] += int(ex.sum() > exK.sum() + 1e-12)
    closure = (tot["K"] - tot["KR"]) / tot["K"] if tot["K"] > 0 else float("nan")
    print(f"[{tag}] excess K={tot['K']:.4f} KR={tot['KR']:.4f} closure={closure:.4f} {cnt}")
    top = np.argsort(-per_col)[:10]
    print(
        f"[{tag}] columns where the reverse step removed excess:",
        [
            (int(j), round(float(per_col[j]), 4), "orig" if j < n_orig else "aux")
            for j in top
            if per_col[j] > 0
        ],
    )
print(f"executed column comparisons: {n_cmp}")
sys.exit(0 if n_cmp else 1)
