"""Offline harness validation with a mock model (no API key needed).

A 'perfect agent' is an INDEPENDENT re-implementation of the task oracle: it reads
its carried running state from the prior assistant message (or the initial state),
applies the single visible instruction, and replies. If the harness is correct, a
perfect agent must score 100% in ALL regimes -- including compressed, which proves
the bounded context window carries enough state. A 'faulty agent' (drops one step)
must yield failures with a correctly located first_error_step.
"""
from __future__ import annotations

import asyncio
import json
import re

import runner
from config import HORIZONS
from runner import extract_json


# ----------------------------------------------------------- instruction apply
def _apply(state: dict, instr: str, target: str | None) -> dict:
    s = dict(state) if isinstance(state, dict) else state
    m = re.match(r"Deposit (\d+) into account (\w)", instr)
    if m:
        s[m.group(2)] = int(s[m.group(2)]) + int(m.group(1)); return s
    m = re.match(r"Withdraw (\d+) from account (\w)", instr)
    if m:
        s[m.group(2)] = int(s[m.group(2)]) - int(m.group(1)); return s
    m = re.match(r"Transfer (\d+) from account (\w) to account (\w)", instr)
    if m:
        a = int(m.group(1)); s[m.group(2)] -= a; s[m.group(3)] += a; return s
    m = re.match(r"(\w+) = (\d+)", instr)              # refchain assignment
    if m:
        if m.group(1) == target:
            s["value"] = int(m.group(2))
        return s
    m = re.match(r"Swap the characters at positions (\d+) and (\d+)", instr)
    if m:
        i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
        cs = list(s["s"]); cs[i], cs[j] = cs[j], cs[i]; s["s"] = "".join(cs); return s
    if instr.startswith("Rotate the string left"):
        s["s"] = s["s"][1:] + s["s"][:1]; return s
    if instr.startswith("Rotate the string right"):
        s["s"] = s["s"][-1:] + s["s"][:-1]; return s
    m = re.match(r"Replace the character at position (\d+) .*with '(\w)'", instr)
    if m:
        i = int(m.group(1)) - 1; cs = list(s["s"]); cs[i] = m.group(2)
        s["s"] = "".join(cs); return s
    raise ValueError(f"mock cannot parse: {instr!r}")


def _target_from_system(sys_text: str) -> str | None:
    m = re.search(r"tracked variable (\w+)", sys_text)
    return m.group(1) if m else None


def _answer_from_state(state: dict, query: str) -> object:
    m = re.search(r"balance of account (\w)", query)
    if m:
        return state[m.group(1)]
    if "final value of" in query:
        return state["value"]
    return state["s"]


def make_mock(faulty_step: int | None = None):
    """Return an async chat() mock. If faulty_step is set, the agent corrupts its
    state exactly once at that step index (to test failure detection)."""

    async def mock_chat(model_key, messages, **gen):
        sys_text = messages[0]["content"]
        target = _target_from_system(sys_text)
        init = json.loads(re.search(r"Initial state: (\{.*\})", sys_text).group(1))

        # carried state = last assistant 'state' in the visible context, else init
        state = init
        for msg in messages:
            if msg["role"] == "assistant":
                p = extract_json(msg["content"])
                if p and "state" in p:
                    state = p["state"]

        last = messages[-1]["content"]
        if last.startswith("Step"):
            idx = int(re.match(r"Step (\d+) of", last).group(1)) - 1
            instr = last.split("Instruction: ", 1)[1].strip()
            state = _apply(state, instr, target)
            if faulty_step is not None and idx == faulty_step:
                # corrupt one numeric/char to force a single wrong step
                if "s" in state:
                    state["s"] = "zzzzz"
                elif "value" in state:
                    state["value"] = -1
                else:
                    state[next(iter(state))] = -999
            content = json.dumps({"state": state})
        elif last.startswith("All"):  # interactive final query
            content = json.dumps({"answer": _answer_from_state(state, last)})
        else:  # padded one-shot: apply every numbered instruction from init
            st = init
            for line in last.splitlines():
                mm = re.match(r"\s*\d+\.\s+(.*)", line)
                if mm:
                    st = _apply(st, mm.group(1).strip(), target)
            content = json.dumps({"answer": _answer_from_state(st, last)})

        return {"content": content, "finish_reason": "stop",
                "prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                "completion_tokens": len(content) // 4, "latency_s": 0.0}

    return mock_chat


async def main() -> None:
    families = ["ledger", "refchain", "cipher"]
    regimes = ["natural", "compressed", "padded"]

    # 1) perfect agent -> 100% everywhere
    runner.chat = make_mock(faulty_step=None)
    fails = 0
    for fam in families:
        for reg in regimes:
            for H in HORIZONS:
                r = await runner.run_trajectory("mock", fam, H, reg, trial=3)
                if not r["success"]:
                    fails += 1
                    print(f"  PERFECT FAIL {fam}/{reg}/H={H}: ans={r['final_answer']} gold={r['gold']}")
                if reg != "padded":
                    assert all(r["per_step_correct"]), f"per-step not all true {fam}/{reg}/H={H}"
    print(f"[perfect] failures={fails} (expect 0)")

    # 2) compressed context really is bounded vs natural growing
    runner.chat = make_mock(faulty_step=None)
    rn = await runner.run_trajectory("mock", "ledger", 16, "natural", trial=1)
    rc = await runner.run_trajectory("mock", "ledger", 16, "compressed", trial=1)
    print(f"[context] natural max_prompt_tokens={rn['max_prompt_tokens']} "
          f"vs compressed={rc['max_prompt_tokens']} (compressed must be << natural)")
    assert rc["max_prompt_tokens"] < rn["max_prompt_tokens"]

    # 3) faulty agent -> failure with first_error at the injected step
    runner.chat = make_mock(faulty_step=2)
    r = await runner.run_trajectory("mock", "cipher", 8, "natural", trial=0)
    print(f"[faulty]  success={r['success']} first_error_step={r['first_error_step']} (expect False / 2)")
    assert r["success"] is False and r["first_error_step"] == 2

    print("test_harness.py: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
