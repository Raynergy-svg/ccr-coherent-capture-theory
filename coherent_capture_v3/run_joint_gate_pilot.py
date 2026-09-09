import csv
import json
from collections import Counter, defaultdict

from joint_gate_screen import iter_pilot_grid

rows = list(iter_pilot_grid())

with open("joint_gate_pilot.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

counts = Counter(r["label"] for r in rows)
by_q = defaultdict(Counter)
min_q_by_trunc = {}
compat = []

for r in rows:
    by_q[r["q_over_ad"]][r["label"]] += 1
    if r["disk_presence_gate"]:
        t = r["truncation_factor"]
        min_q_by_trunc[t] = min(min_q_by_trunc.get(t, float("inf")), r["q_over_ad"])
    if r["label"] == "SCREEN_COMPATIBLE":
        compat.append(r)

qvals = sorted(set(r["q_over_ad"] for r in rows))
fsvals = sorted(set(r["f_sigma"] for r in rows))
lvals = sorted(set(r["remaining_myr"] for r in rows))
compat_keys = {
    (r["v_inf_kms"], r["truncation_factor"], r["e_i"], r["i_i_deg"],
     r["q_over_ad"], r["f_sigma"], r["remaining_myr"])
    for r in compat
}

contiguous = False
example = None
for vinf in sorted(set(r["v_inf_kms"] for r in rows)):
    for trunc in sorted(set(r["truncation_factor"] for r in rows)):
        for e_i in sorted(set(r["e_i"] for r in rows)):
            for inc in sorted(set(r["i_i_deg"] for r in rows)):
                for iq in range(len(qvals) - 1):
                    for jf in range(len(fsvals) - 1):
                        for kl in range(len(lvals) - 1):
                            cube = [
                                (vinf, trunc, e_i, inc, q, fs, life)
                                for q in qvals[iq:iq + 2]
                                for fs in fsvals[jf:jf + 2]
                                for life in lvals[kl:kl + 2]
                            ]
                            if all(k in compat_keys for k in cube):
                                contiguous = True
                                example = {
                                    "v_inf_kms": vinf,
                                    "truncation_factor": trunc,
                                    "e_i": e_i,
                                    "i_i_deg": inc,
                                    "q_pair": qvals[iq:iq + 2],
                                    "f_sigma_pair": fsvals[jf:jf + 2],
                                    "lifetime_pair": lvals[kl:kl + 2],
                                }
                                break
                        if contiguous:
                            break
                    if contiguous:
                        break
                if contiguous:
                    break
            if contiguous:
                break
        if contiguous:
            break
    if contiguous:
        break

summary = {
    "n_rows": len(rows),
    "counts": dict(counts),
    "fractions": {k: v / len(rows) for k, v in counts.items()},
    "min_q_over_ad_with_disk_presence_by_truncation": {
        str(k): v for k, v in min_q_by_trunc.items()
    },
    "compatible_count": len(compat),
    "max_compatible_e": max((r["e_i"] for r in compat), default=None),
    "max_compatible_i_deg": max((r["i_i_deg"] for r in compat), default=None),
    "min_compatible_q_over_ad": min((r["q_over_ad"] for r in compat), default=None),
    "contiguous_screen_region": contiguous,
    "contiguous_example": example,
    "by_q": {str(q): dict(c) for q, c in sorted(by_q.items())},
}

with open("joint_gate_pilot_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
