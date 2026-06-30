"""Trajectory runner: drives one agent episode under a given regime, scores it,
and checkpoints results. Trajectories are independent and run concurrently; the
per-call concurrency cap lives in llm.py.

Result schema (one JSON line per trajectory in results/raw/<model>__<family>__<regime>.jsonl):
  model, family, horizon, regime, trial, seed,
  success (bool), final_answer, gold,
  n_turns, per_step_correct (list[bool]), first_error_step (int|null),
  malformed_count, mean_prompt_tokens, max_prompt_tokens,
  completion_tokens, finish_reasons (list), latency_s
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

from config import (FULL_TRIALS, HORIZONS, MAX_COMPLETION_TOKENS, MAX_TURNS_SLACK,
                    MODELS, PILOT_TRIALS, RAW_DIR, REGIMES, TASK_FAMILIES,
                    TEMPERATURE, TOP_P)
from llm import METER, chat
from tasks import AGENTIC_FAMILIES, make_task

_WRITE_LOCK = asyncio.Lock()


# --------------------------------------------------------------- JSON parsing
def _all_json_objects(text: str) -> list[dict]:
    """Every balanced {...} object in the text (tolerates markdown fences, prose,
    and MULTIPLE concatenated objects — some models emit one JSON per line)."""
    objs = []
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, end = 0, False, False, -1
        for k in range(start, len(text)):
            c = text[k]
            if in_str:
                esc = (c == "\\" and not esc)
                if c == '"' and not esc:
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end == -1:
            break
        try:
            o = json.loads(text[start:end + 1])
            if isinstance(o, dict):
                objs.append(o)
        except json.JSONDecodeError:
            pass
        start = text.find("{", end + 1)
    return objs


def extract_json(text: str, want_key: str | None = None) -> dict | None:
    """Pull a JSON object out of a model reply. If want_key is given, return the
    LAST object that contains it (models that replay the whole exchange end with
    the operative object); otherwise the first object. None if unparseable."""
    if not text:
        return None
    objs = _all_json_objects(text)
    if not objs:
        return None
    if want_key is not None:
        keyed = [o for o in objs if want_key in o]
        if keyed:
            return keyed[-1]
    return objs[0]


# --------------------------------------------------------------- context policy
def _context(history: list[dict], compressed: bool) -> list[dict]:
    """Messages actually sent to the model this turn.
    natural  -> full history.
    compressed -> system + last 2 non-system messages (carried state + new obs).
    """
    if not compressed:
        return history
    system = history[0]
    tail = history[1:]
    return [system] + tail[-2:]


# --------------------------------------------------------------- one trajectory
async def run_trajectory(model_key: str, family: str, horizon: int, regime: str,
                         trial: int, temperature: float | None = None) -> dict:
    task = make_task(family, horizon, trial)
    rec = {"model": model_key, "family": family, "horizon": horizon,
           "regime": regime, "trial": trial, "seed": task.seed}
    gen = dict(temperature=TEMPERATURE if temperature is None else temperature,
               top_p=TOP_P, max_tokens=MAX_COMPLETION_TOKENS, seed=task.seed)

    if family in AGENTIC_FAMILIES:
        return await _run_agentic(task, model_key, rec, gen)
    if regime == "padded":
        return await _run_oneshot(task, model_key, rec, gen)
    return await _run_interactive(task, model_key, rec, gen,
                                  compressed=(regime == "compressed"))


async def _run_agentic(task, model_key, rec, gen) -> dict:
    """Agent-driven ReAct loop: the model chooses tool calls until it answers."""
    messages = [{"role": "system", "content": task.system_prompt()},
                {"role": "user", "content": task.first_user()}]
    tool_calls, malformed, n_turns, latency = 0, 0, 0, 0.0
    prompt_toks, finish, compl_toks = [], [], 0
    success, ans = False, None

    for _ in range(task.max_turns):
        out = await chat(model_key, messages, **gen)
        messages.append({"role": "assistant", "content": out["content"]})
        prompt_toks.append(out["prompt_tokens"]); compl_toks += out["completion_tokens"]
        finish.append(out["finish_reason"]); latency += out.get("latency_s", 0.0)
        n_turns += 1

        action = extract_json(out["content"], "answer") or extract_json(out["content"], "tool")
        if action is None:
            malformed += 1
            messages.append({"role": "user", "content":
                "Invalid format. Reply with exactly one JSON object: a tool call or an answer."})
            continue
        obs, done, ok = task.tool_response(action)
        if done:
            ans = action.get("answer"); success = ok; break
        tool_calls += 1
        messages.append({"role": "user", "content": obs})

    rec.update(success=success, final_answer=ans, gold=task.answer, n_turns=n_turns,
               per_step_correct=[], first_error_step=None, malformed_count=malformed,
               tool_calls=tool_calls,
               mean_prompt_tokens=round(sum(prompt_toks) / len(prompt_toks), 1) if prompt_toks else 0,
               max_prompt_tokens=max(prompt_toks) if prompt_toks else 0,
               completion_tokens=compl_toks, finish_reasons=finish,
               latency_s=round(latency, 2))
    return rec


async def _run_interactive(task, model_key, rec, gen, *, compressed: bool) -> dict:
    history = [{"role": "system", "content": task.system_prompt()}]
    per_step, prompt_toks, finish, compl_toks = [], [], [], 0
    malformed, first_error, latency = 0, None, 0.0

    for i in range(task.horizon):
        history.append({"role": "user", "content": task.step_observation(i)})
        out = await chat(model_key, _context(history, compressed), **gen)
        history.append({"role": "assistant", "content": out["content"]})
        prompt_toks.append(out["prompt_tokens"]); compl_toks += out["completion_tokens"]
        finish.append(out["finish_reason"]); latency += out.get("latency_s", 0.0)

        parsed = extract_json(out["content"], "state")
        if not parsed or "state" not in parsed:
            malformed += 1
            ok = False
        else:
            ok = task.check_state(i, parsed["state"])
        per_step.append(ok)
        if not ok and first_error is None:
            first_error = i

    # final query
    history.append({"role": "user", "content": task.final_observation()})
    out = await chat(model_key, _context(history, compressed), **gen)
    prompt_toks.append(out["prompt_tokens"]); compl_toks += out["completion_tokens"]
    finish.append(out["finish_reason"]); latency += out.get("latency_s", 0.0)
    parsed = extract_json(out["content"], "answer")
    ans = parsed.get("answer") if parsed else None
    if parsed is None or "answer" not in parsed:
        malformed += 1
    success = task.check_answer(ans) if ans is not None else False

    rec.update(success=success, final_answer=ans, gold=task.answer,
               n_turns=task.horizon + 1, per_step_correct=per_step,
               first_error_step=first_error, malformed_count=malformed,
               mean_prompt_tokens=round(sum(prompt_toks) / len(prompt_toks), 1),
               max_prompt_tokens=max(prompt_toks), completion_tokens=compl_toks,
               finish_reasons=finish, latency_s=round(latency, 2))
    return rec


async def _run_oneshot(task, model_key, rec, gen) -> dict:
    messages = [{"role": "system", "content": task.system_prompt()},
                {"role": "user", "content": task.oneshot_prompt()}]
    out = await chat(model_key, messages, **gen)
    parsed = extract_json(out["content"], "answer")
    ans = parsed.get("answer") if parsed else None
    malformed = 0 if (parsed and "answer" in parsed) else 1
    success = task.check_answer(ans) if ans is not None else False
    rec.update(success=success, final_answer=ans, gold=task.answer,
               n_turns=1, per_step_correct=[], first_error_step=None,
               malformed_count=malformed,
               mean_prompt_tokens=out["prompt_tokens"],
               max_prompt_tokens=out["prompt_tokens"],
               completion_tokens=out["completion_tokens"],
               finish_reasons=[out["finish_reason"]],
               latency_s=out.get("latency_s", 0.0))
    return rec


# --------------------------------------------------------------- checkpointing
def _raw_path(model_key: str, family: str, regime: str) -> Path:
    return RAW_DIR / f"{model_key}__{family}__{regime}.jsonl"


def _completed(model_key: str, family: str, regime: str) -> set[tuple[int, int]]:
    p = _raw_path(model_key, family, regime)
    done = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["horizon"], r["trial"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def _run_and_save(model_key, family, horizon, regime, trial) -> dict | None:
    # Isolate failures: a trajectory that errors after retries is skipped (not written),
    # so it is retried on the next resume and never kills the whole sweep.
    try:
        rec = await run_trajectory(model_key, family, horizon, regime, trial)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {model_key}/{family}/{regime}/H{horizon}/t{trial}: {str(e)[:80]}")
        return None
    async with _WRITE_LOCK:
        with _raw_path(model_key, family, regime).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# --------------------------------------------------------------- sweep
async def sweep(models: list[str], families: list[str], horizons: list[int],
                regimes: list[str], trials: int, resume: bool = True) -> None:
    jobs = []
    for m in models:
        for fam in families:
            for reg in regimes:
                done = _completed(m, fam, reg) if resume else set()
                for h in horizons:
                    for t in range(trials):
                        if (h, t) in done:
                            continue
                        jobs.append(_run_and_save(m, fam, h, reg, t))
    print(f"Scheduling {len(jobs)} trajectories "
          f"(models={models}, families={families}, horizons={horizons}, "
          f"regimes={regimes}, trials={trials}).")
    if not jobs:
        print("Nothing to do — all cells already complete.")
        return
    await tqdm_asyncio.gather(*jobs)
    print("Done. " + METER.summary())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--families", nargs="*", default=TASK_FAMILIES)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--regimes", nargs="*", default=REGIMES)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="use PILOT_TRIALS")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    trials = args.trials if args.trials is not None else (
        PILOT_TRIALS if args.pilot else FULL_TRIALS)
    asyncio.run(sweep(args.models, args.families, args.horizons, args.regimes,
                      trials, resume=not args.no_resume))


if __name__ == "__main__":
    main()
