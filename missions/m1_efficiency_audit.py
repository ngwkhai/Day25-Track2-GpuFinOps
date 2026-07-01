"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension: MBU-based right-sizing for memory-bound GPUs
    vram_cost = {
        g: num(catalog_by_type()[g]["on_demand_hr"]) / num(catalog_by_type()[g]["hbm_gb"])
        for g in cat
    }
    rightsizing = []
    for s in summary:
        if s["mbu"] >= 0.35:
            continue
        cur = s["gpu_type"]
        cur_bw = num(cat[cur]["peak_bw_tbs"])
        candidates = [
            g for g in cat
            if g != cur
            and num(cat[g]["peak_bw_tbs"]) >= cur_bw * 0.9
            and num(cat[g]["on_demand_hr"]) < num(cat[cur]["on_demand_hr"])
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda g: num(cat[g]["on_demand_hr"]))
        monthly_save = (num(cat[cur]["on_demand_hr"]) - num(cat[best]["on_demand_hr"])) * 24 * 30
        rightsizing.append({
            "gpu_id": s["gpu_id"], "current": cur, "suggested": best,
            "mbu": s["mbu"], "monthly_save": round(monthly_save, 0),
            "vram_cost_cur": round(vram_cost[cur], 4), "vram_cost_new": round(vram_cost[best], 4),
        })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")
        if rightsizing:
            print("\nMBU right-sizing (memory-bound, MBU<35%):")
            for r in rightsizing:
                print(f"  {r['gpu_id']:14} {r['current']} -> {r['suggested']}  "
                      f"MBU={r['mbu']:.2f}  save ${r['monthly_save']:,.0f}/mo  "
                      f"($/GB-VRAM {r['vram_cost_cur']:.4f} -> {r['vram_cost_new']:.4f})")

    return {
        "summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
        "rightsizing": rightsizing,
    }


if __name__ == "__main__":
    run()
