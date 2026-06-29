"""Synthetic, oracle-verifiable agent task families with a parametric horizon.

Each task instance is defined by (family, horizon H, trial seed) and is presented
under one of three regimes by the runner (natural / compressed / padded). The
SAME underlying instance is reused across regimes so regime comparisons are paired.

Protocol (interactive regimes natural & compressed):
  - The environment streams H instructions, one per turn.
  - Each turn the agent must reply with a single JSON object carrying its updated
    running STATE (schema is family-specific). No correctness feedback is given
    (so per-step errors can compound and be measured).
  - After the H-th instruction the environment asks the final query; the agent
    replies with {"answer": ...}.

Protocol (padded regime):
  - All H instructions are delivered in one long prompt; the agent answers in a
    single turn (turns=1, context long). Isolates context-length from step-count.

Each family precomputes, deterministically from the seed:
  - instructions : list[str]      (length H)
  - states       : list[state]    (oracle running state AFTER each instruction)
  - query        : str            (the final question)
  - answer       : the gold answer (compared to the agent's "answer")
  - state_schema : a short human description of the per-turn STATE object
  - initial_state: the oracle state before any instruction
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------- base
@dataclass
class Task:
    family: str
    horizon: int
    seed: int
    instructions: list[str] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    initial_state: Any = None
    query: str = ""
    answer: Any = None
    state_schema: str = ""
    goal_desc: str = ""

    # ---- formatting shared by all families ----
    def system_prompt(self) -> str:
        return (
            f"You are a careful agent solving a {self.goal_desc} task over "
            f"multiple steps.\n"
            f"At each step you receive one instruction. You must update and report "
            f"your running state.\n\n"
            f"STATE FORMAT: {self.state_schema}\n\n"
            f"On every step, reply with EXACTLY ONE JSON object and nothing else, of "
            f'the form: {{"state": <STATE>}}. '
            f"Keep your state accurate; earlier mistakes are not corrected for you.\n"
            f'When asked the final question, reply with EXACTLY ONE JSON object '
            f'{{"answer": <VALUE>}} and nothing else.\n'
            f"Initial state: {json.dumps(self.initial_state)}"
        )

    def step_observation(self, i: int) -> str:
        return f"Step {i + 1} of {self.horizon}. Instruction: {self.instructions[i]}"

    def final_observation(self) -> str:
        return f"All {self.horizon} instructions done. {self.query} " \
               f'Reply with {{"answer": <VALUE>}}.'

    def oneshot_prompt(self) -> str:
        lines = [f"You will be given {self.horizon} instructions to apply in order, "
                 f"starting from the initial state, then a question.",
                 f"Initial state: {json.dumps(self.initial_state)}",
                 f"STATE MEANING: {self.state_schema}", "", "Instructions:"]
        for i, ins in enumerate(self.instructions):
            lines.append(f"{i + 1}. {ins}")
        lines += ["", self.query,
                  'Reply with EXACTLY ONE JSON object {"answer": <VALUE>} and nothing else.']
        return "\n".join(lines)

    # ---- grading ----
    def check_answer(self, ans: Any) -> bool:
        return self._norm(ans) == self._norm(self.answer)

    def check_state(self, i: int, submitted: Any) -> bool:
        """Is the agent's submitted state correct after instruction i?"""
        return self._norm_state(submitted) == self._norm_state(self.states[i])

    # overridable normalizers
    def _norm(self, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    def _norm_state(self, s: Any) -> Any:
        return s


# ----------------------------------------------------------------- ledger
class LedgerTask(Task):
    """Maintain 3 account balances through H transactions; report a queried one."""
    ACCOUNTS = ["A", "B", "C"]

    @classmethod
    def generate(cls, horizon: int, seed: int) -> "LedgerTask":
        rng = random.Random(seed)
        start = 100
        bal = {a: start for a in cls.ACCOUNTS}
        instructions, states = [], []
        for _ in range(horizon):
            kind = rng.choice(["deposit", "withdraw", "transfer"])
            if kind == "transfer":
                src, dst = rng.sample(cls.ACCOUNTS, 2)
                amt = rng.randint(1, 50)
                bal[src] -= amt
                bal[dst] += amt
                instructions.append(f"Transfer {amt} from account {src} to account {dst}.")
            elif kind == "deposit":
                acc = rng.choice(cls.ACCOUNTS)
                amt = rng.randint(1, 50)
                bal[acc] += amt
                instructions.append(f"Deposit {amt} into account {acc}.")
            else:
                acc = rng.choice(cls.ACCOUNTS)
                amt = rng.randint(1, 50)
                bal[acc] -= amt
                instructions.append(f"Withdraw {amt} from account {acc}.")
            states.append(dict(bal))
        queried = rng.choice(cls.ACCOUNTS)
        return cls(
            family="ledger", horizon=horizon, seed=seed,
            instructions=instructions, states=states,
            initial_state={a: start for a in cls.ACCOUNTS},
            query=f"What is the final balance of account {queried}?",
            answer=bal[queried],
            state_schema='an object mapping each account "A","B","C" to its integer '
                         'balance, e.g. {"A": 120, "B": 95, "C": 100}',
            goal_desc="bank-ledger accounting",
        )

    def _norm_state(self, s: Any) -> Any:
        if not isinstance(s, dict):
            return s
        try:
            return {k: int(s.get(k)) for k in self.ACCOUNTS}
        except (TypeError, ValueError):
            return s


# ----------------------------------------------------------------- refchain
class RefchainTask(Task):
    """Track the current binding of one announced variable through H assignments,
    among distractor variables and several reassignments of the target."""

    @classmethod
    def generate(cls, horizon: int, seed: int) -> "RefchainTask":
        rng = random.Random(seed)
        n_vars = max(5, horizon)
        names = [f"X{i}" for i in range(n_vars)]
        target = rng.choice(names)

        # choose 2-4 reassignment positions for the target, spread across the run,
        # guaranteeing at least one and a controlled "last" position.
        n_reassign = min(max(2, horizon // 4), horizon)
        positions = sorted(rng.sample(range(horizon), n_reassign))
        # ensure the FIRST instruction binds the target so it always has a value
        if 0 not in positions:
            positions[0] = 0
            positions = sorted(set(positions))

        instructions, states = [], []
        cur = None
        for i in range(horizon):
            if i in positions:
                val = rng.randint(10, 999)
                instructions.append(f"{target} = {val}")
                cur = val
            else:
                d = rng.choice([n for n in names if n != target])
                val = rng.randint(10, 999)
                instructions.append(f"{d} = {val}")
            states.append({"value": cur})

        return cls(
            family="refchain", horizon=horizon, seed=seed,
            instructions=instructions, states=states,
            initial_state={"value": None},
            query=f"What is the final value of {target}?",
            answer=cur,
            state_schema=f'an object {{"value": <int>}} giving the current value of '
                         f'the tracked variable {target} (ignore all other variables)',
            goal_desc="variable-tracking",
        )

    def _norm_state(self, s: Any) -> Any:
        if not isinstance(s, dict):
            return s
        try:
            return {"value": int(s.get("value"))}
        except (TypeError, ValueError):
            return {"value": s.get("value")}


# ----------------------------------------------------------------- cipher
class CipherTask(Task):
    """Apply H string-edit operations to a length-5 string; report the result."""
    LEN = 5

    @classmethod
    def generate(cls, horizon: int, seed: int) -> "CipherTask":
        rng = random.Random(seed)
        alpha = "abcdefghijklmnopqrstuvwxyz"
        s = [rng.choice(alpha) for _ in range(cls.LEN)]
        start = "".join(s)
        instructions, states = [], []
        for _ in range(horizon):
            op = rng.choice(["swap", "rotL", "rotR", "replace"])
            if op == "swap":
                i, j = rng.sample(range(cls.LEN), 2)
                s[i], s[j] = s[j], s[i]
                instructions.append(f"Swap the characters at positions {i + 1} and {j + 1} (1-indexed).")
            elif op == "rotL":
                s = s[1:] + s[:1]
                instructions.append("Rotate the string left by one (move the first character to the end).")
            elif op == "rotR":
                s = s[-1:] + s[:-1]
                instructions.append("Rotate the string right by one (move the last character to the front).")
            else:
                i = rng.randrange(cls.LEN)
                c = rng.choice(alpha)
                s[i] = c
                instructions.append(f"Replace the character at position {i + 1} (1-indexed) with '{c}'.")
            states.append({"s": "".join(s)})
        return cls(
            family="cipher", horizon=horizon, seed=seed,
            instructions=instructions, states=states,
            initial_state={"s": start},
            query="What is the final string?",
            answer="".join(s),
            state_schema='an object {"s": "<5-char string>"} giving the current string',
            goal_desc="string-manipulation",
        )

    def _norm(self, v: Any) -> Any:
        if isinstance(v, dict) and "s" in v:
            v = v["s"]
        return str(v).strip().lower()

    def _norm_state(self, s: Any) -> Any:
        if isinstance(s, dict) and "s" in s:
            return str(s["s"]).strip().lower()
        return str(s).strip().lower()


# ----------------------------------------------------------------- registry
_FAMILIES = {"ledger": LedgerTask, "refchain": RefchainTask, "cipher": CipherTask}


def make_task(family: str, horizon: int, trial: int) -> Task:
    """Deterministic instance for (family, horizon, trial), regime-independent."""
    seed = (hash((family, horizon, trial)) & 0x7FFFFFFF)
    return _FAMILIES[family].generate(horizon, seed)


if __name__ == "__main__":  # quick self-test of the oracles
    for fam in _FAMILIES:
        for H in (2, 4, 8):
            t = make_task(fam, H, 0)
            assert len(t.instructions) == H == len(t.states)
            assert t.check_answer(t.answer)
            assert t.check_state(H - 1, t.states[-1])
            print(f"{fam} H={H}: {t.instructions[0]!r} ... Q={t.query!r} A={t.answer!r}")
    print("tasks.py self-test OK")
