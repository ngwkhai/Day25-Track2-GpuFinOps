"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_WRITE_COST_PER_M = 0.05  # illustrative storage/write cost per 1M cached tokens


def _reasoning_stats(rows) -> dict:
    reasoning = [r for r in rows if int(num(r["is_reasoning"]))]
    normal = [r for r in rows if not int(num(r["is_reasoning"]))]
    r_cost = n_cost = r_wh = n_wh = r_tokens = n_tokens = 0
    for r in reasoning:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        r_cost += pricing.request_cost(inp, out, pin, pout,
                                       cached_in=int(num(r["cached_input_tokens"])),
                                       batch=bool(int(num(r["is_batch"]))))
        r_wh += sustainability.wh_per_query(inp + out, is_reasoning=True)
        r_tokens += inp + out
    for r in normal:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        n_cost += pricing.request_cost(inp, out, pin, pout,
                                       cached_in=int(num(r["cached_input_tokens"])),
                                       batch=bool(int(num(r["is_batch"]))))
        n_wh += sustainability.wh_per_query(inp + out, is_reasoning=False)
        n_tokens += inp + out
    total_cost = r_cost + n_cost
    total_wh = r_wh + n_wh
    return {
        "reasoning_requests": len(reasoning),
        "normal_requests": len(normal),
        "reasoning_cost": round(r_cost, 2),
        "normal_cost": round(n_cost, 2),
        "reasoning_cost_pct": round(r_cost / total_cost * 100, 1) if total_cost else 0.0,
        "reasoning_traffic_pct": round(len(reasoning) / len(rows) * 100, 1) if rows else 0.0,
        "reasoning_wh": round(r_wh, 2),
        "normal_wh": round(n_wh, 2),
        "reasoning_wh_pct": round(r_wh / total_wh * 100, 1) if total_wh else 0.0,
        "reasoning_tokens": r_tokens,
        "normal_tokens": n_tokens,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_writes = sum(1 for r in rows if int(num(r["cached_input_tokens"])) > 0)
    # proxy: each cached request reuses its prefix ~3x on average in this workload
    avg_reads = 3.0 if cache_writes else 0.0
    cache_ok = pricing.cache_is_worth_it(avg_reads, CACHE_WRITE_COST_PER_M)

    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(
            inp, out, pin, pout,
            cached_in=cached if cache_ok else 0,
            batch=is_batch,
        )

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    reasoning = _reasoning_stats(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        be_reads = pricing.cache_break_even_reads(CACHE_WRITE_COST_PER_M)
        print(f"\nCache economics: avg_reads={avg_reads:.1f}  break-even={be_reads:.2f}  worth_it={cache_ok}")
        print(f"Reasoning: {reasoning['reasoning_requests']} reqs ({reasoning['reasoning_traffic_pct']}% traffic) "
              f"-> ${reasoning['reasoning_cost']:.2f} ({reasoning['reasoning_cost_pct']}% cost), "
              f"{reasoning['reasoning_wh']:.1f} Wh ({reasoning['reasoning_wh_pct']}% energy)")
        print(f"  -> Rule: cap reasoning to low-confidence tasks only (currently ~{reasoning['reasoning_wh_pct']}% of Wh)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_worth_it": cache_ok, "avg_cache_reads": round(avg_reads, 2),
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
