"""Temperature-sensitivity sub-study: re-run one model x one family across a small
grid of temperatures to check the decay result is not an artifact of the fixed
T=0.2 setting. Writes results/temp_study.jsonl and prints success by temp x horizon.

Run:  AGENT_ROT_BACKEND=openrouter python temp_study.py
"""
from __future__ import annotations

import asyncio
import json

from config import HORIZONS, RESULTS_DIR
from llm import METER
from runner import run_trajectory

MODEL = "qwen7b"        # a clean, cheap, mid-capability model
FAMILY = "ledger"       # the family with the clearest decay signal
TEMPS = [0.0, 0.5, 1.0]
TRIALS = 10
OUT = RESULTS_DIR / "temp_study.jsonl"


async def main() -> None:
    jobs = []
    for temp in TEMPS:
        for h in HORIZONS:
            for t in range(TRIALS):
                jobs.append((temp, h, t))

    async def one(temp, h, t):
        r = await run_trajectory(MODEL, FAMILY, h, "natural", t, temperature=temp)
        r["temperature"] = temp
        return r

    results = await asyncio.gather(*[one(*j) for j in jobs])
    with OUT.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summarize success by temperature x horizon
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for r in results:
        k = (r["temperature"], r["horizon"])
        agg[k][0] += int(r["success"]); agg[k][1] += 1
    print(f"{MODEL} / {FAMILY} — success by temperature x horizon ({TRIALS} trials):")
    print("temp \\ H   " + "  ".join(f"{h:>4}" for h in HORIZONS))
    for temp in TEMPS:
        row = "  ".join(f"{agg[(temp,h)][0]/agg[(temp,h)][1]:>4.2f}" for h in HORIZONS)
        print(f"  {temp:<6}  {row}")
    print(METER.summary())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
