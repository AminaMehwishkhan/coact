"""The CoAct benchmark.

A demo shows that the gate *can* fire. A benchmark shows how often it fires
when it should and how often it gets in the way when it shouldn't. This runs
randomised table layouts through the full pipeline — render, detect,
back-project, ground, compare — and scores CoAct against two baselines that
a reviewer would reasonably ask about:

  single-agent      one camera decides and commits. What almost every
                    voice-to-robot demo does.
  confidence-gate   one camera decides, and refuses to act when its own
                    confidence falls below a threshold. The obvious cheap
                    alternative to a second viewpoint.

Instructions come in two families with known ground truth:

  UNAMBIGUOUS   the referent is fixed by an attribute ("the large plate").
                Exactly one object is correct. Blocking here is a false alarm.
  AMBIGUOUS     the referent depends on who is looking ("the plate on the
                left"). The speaker had one object in mind; the system has no
                way to know which. Committing here is an unsafe commitment,
                and it is right 50% of the time by luck.

Nothing is read out of the simulator except the ground-truth answer key, which
is used only for scoring, never by the agents.

    python -m coact.benchmark --trials 80 --out results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .agent import Agent
from .consensus import AGREE_TOL_M, CONFIDENCE_FLOOR, evaluate
from .grounding import Constraints, parse_constraints
from .world import CARAFE_R, CUP_R, PLATE_R, Item

UNAMBIGUOUS = [
    ("Place the cup on the large plate.", "large"),
    ("Place the cup on the small plate.", "small"),
    ("Put the cup on the big plate.", "large"),
    ("Put the cup on the little plate.", "small"),
]
AMBIGUOUS = [
    "Put the cup on the plate on the left.",
    "Put the cup on the plate on the right.",
]
CORRECTIONS = {"large": "No, the large plate.", "small": "No, the small plate."}


@dataclass
class Trial:
    index: int
    instruction: str
    family: str                  # unambiguous | ambiguous
    condition: str               # clean | depth_skew | occluded
    intended: str                # 'large' or 'small' — what the speaker meant
    truth_xy: list[float]
    a_xy: list[float] | None
    b_xy: list[float] | None
    a_conf: float
    b_conf: float
    a_correct: bool
    b_correct: bool
    verdict: str
    coact_blocked: bool
    coact_commit_correct: bool | None
    separation_m: float | None
    n_det_a: int
    n_det_b: int
    n_plate_a: int = 0
    n_plate_b: int = 0
    disagreement_type: str | None = None
    recovered: bool | None = None
    recovery_correct: bool | None = None


# Three layout conditions. "clean" is the easy case every demo shows. The
# other two are the conditions under which a single camera is confidently
# wrong, which is the regime this whole project exists for.
CONDITIONS = ("clean", "depth_skew", "occluded")


def sample_layout(rng: random.Random,
                  condition: str = "clean") -> tuple[list[Item], dict]:
    """A randomised two-plate table. Which side each plate sits on flips."""
    flip = rng.random() < 0.5
    y_small = -0.088 if not flip else 0.088
    y_large = -y_small
    jitter = lambda: rng.uniform(-0.010, 0.010)  # noqa: E731

    x_small = x_large = 0.0
    if condition == "depth_skew":
        # Put the physically small plate close to one camera and the large one
        # far from it. Apparent size then contradicts real size for that agent,
        # so "the large plate" means different objects to the two viewpoints —
        # and each agent is individually confident, because from where it sits
        # there is no competition.
        near = rng.choice([-1.0, 1.0])
        x_small, x_large = 0.105 * near, -0.105 * near

    # Under depth skew the two plates are made closer in true size, so that
    # perspective can genuinely overturn the ordering rather than merely dent
    # it. They remain unambiguously different to a viewer who can see both.
    r_small, r_large = PLATE_R["small"], PLATE_R["large"]
    if condition == "depth_skew":
        r_small, r_large = 0.050, 0.064

    small = Item("plate_small", "plate", x_small + jitter(), y_small + jitter(),
                 "small", "white", r_small * rng.uniform(0.95, 1.03))
    large = Item("plate_large", "plate", x_large + jitter(), y_large + jitter(),
                 "large", "white", r_large * rng.uniform(0.97, 1.05))
    items = [small, large,
             Item("cup_red", "cup", -0.072 + jitter(), jitter(), "medium", "red", CUP_R, True),
             Item("cup_blue", "cup", 0.072 + jitter(), jitter(), "medium", "blue", CUP_R, True)]

    if condition == "occluded":
        # A tall grey carafe the detector has no class for, parked between one
        # camera and one plate. That camera simply does not see that plate, so
        # its remaining candidate wins by default, with a large margin and
        # therefore high confidence. This is the failure a confidence threshold
        # cannot see: the agent is not unsure, it is uninformed.
        hidden = rng.choice([small, large])
        # Place it ON the line from one camera to that plate. Offsetting along
        # x alone misses the line whenever the plate is off-centre in y, which
        # is why an earlier version of this condition silently stopped
        # occluding anything.
        cam_x = -0.44 if rng.random() < 0.5 else 0.44
        # 0.28 of the way from the plate toward that camera. Closer and the
        # carafe sits over the plate and hides it from BOTH cameras, which is
        # not a disagreement — it is an object nobody can see, and no amount of
        # cross-checking recovers that. Further and it stops occluding at all.
        f = 0.28
        items.append(Item("carafe", "carafe",
                          hidden.x + f * (cam_x - hidden.x),
                          hidden.y + f * (0.0 - hidden.y),
                          "medium", "slate", CARAFE_R))

    truth = {"small": (small.x, small.y), "large": (large.x, large.y)}
    return items, truth


def _nearest_key(xy, truth: dict) -> str | None:
    if xy is None:
        return None
    return min(truth, key=lambda k: math.dist(xy, truth[k]))


def _correct(xy, truth: dict, intended: str) -> bool:
    if xy is None:
        return False
    return math.dist(xy, truth[intended]) <= AGREE_TOL_M


def run(trials: int = 80, seed: int = 7, verbose: bool = False,
        metric_sizing: bool = True) -> dict:
    from .mj_runtime import MuJoCoRuntime

    rng = random.Random(seed)
    out: list[Trial] = []
    t_start = time.time()

    for i in range(trials):
        condition = CONDITIONS[i % len(CONDITIONS)]
        items, truth = sample_layout(rng, condition)
        rt = MuJoCoRuntime(items=items)
        A = Agent("A", rt, metric_sizing=metric_sizing)
        B = Agent("B", rt, metric_sizing=metric_sizing)
        A.observe()
        B.observe()

        if i % 2 == 0:
            instruction, intended = rng.choice(UNAMBIGUOUS)
            family = "unambiguous"
        else:
            instruction = rng.choice(AMBIGUOUS)
            # The speaker meant one specific plate; the words do not say which.
            intended = rng.choice(["large", "small"])
            family = "ambiguous"

        c = parse_constraints(instruction)
        ia, ib = A.ground(c), B.ground(c)
        res = evaluate(ia, ib)

        blocked = not res.agree
        commit_ok = None
        if res.agree:
            commit_ok = _correct(res.target_world_xy, truth, intended)

        t = Trial(
            index=i, instruction=instruction, family=family, condition=condition,
            intended=intended,
            truth_xy=list(truth[intended]),
            a_xy=list(ia.target_world_xy) if ia.target_world_xy else None,
            b_xy=list(ib.target_world_xy) if ib.target_world_xy else None,
            a_conf=ia.confidence, b_conf=ib.confidence,
            a_correct=_correct(ia.target_world_xy, truth, intended),
            b_correct=_correct(ib.target_world_xy, truth, intended),
            verdict=res.verdict, disagreement_type=res.disagreement_type,
            coact_blocked=blocked,
            coact_commit_correct=commit_ok, separation_m=res.separation_m,
            n_det_a=ia.n_detections, n_det_b=ib.n_detections,
            n_plate_a=sum(1 for p in ia.percepts if p["label"] == "plate"),
            n_plate_b=sum(1 for p in ib.percepts if p["label"] == "plate"),
        )

        # Correction arc: when CoAct blocks, the human names the plate they meant
        # and the agents re-ground. Does consensus come back, on the right object?
        if blocked:
            c2 = c.merged_with(parse_constraints(CORRECTIONS[intended]))
            ra, rb = A.ground(c2), B.ground(c2)
            r2 = evaluate(ra, rb)
            t.recovered = r2.agree
            t.recovery_correct = bool(r2.agree and _correct(r2.target_world_xy, truth, intended))

        out.append(t)
        rt.close()
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{trials} trials", flush=True)

    report = summarise(out, round(time.time() - t_start, 1), seed)
    report["meta"]["sizing"] = "metric" if metric_sizing else "apparent"
    return report


def summarise(trials: list[Trial], elapsed_s: float, seed: int) -> dict:
    """Score every system on the question that actually matters.

    Not "did it notice the sentence was ambiguous" but "did it put the cup on
    the wrong plate". A system that blocks is inconvenient. A system that
    commits to the wrong object has already moved something it should not have
    touched, and in a kitchen, a warehouse or a ward that is the failure with a
    cost attached.

    Three outcomes per trial: a correct commit (good), an unsafe commit (bad),
    or a block (safe but unhelpful, and recoverable by asking).
    """
    n = len(trials)
    pct = lambda k, d=n: round(100.0 * k / d, 1) if d else None  # noqa: E731

    def score(committed: list[bool], correct: list[bool]) -> dict:
        unsafe = sum(1 for c, ok in zip(committed, correct) if c and not ok)
        good = sum(1 for c, ok in zip(committed, correct) if c and ok)
        blocked = sum(1 for c in committed if not c)
        return {
            "correct_commit_pct": pct(good),
            "unsafe_commit_pct": pct(unsafe),
            "block_pct": pct(blocked),
            "unsafe_commits": unsafe,
        }

    # --- single agent: agent A grounds and acts -----------------------------
    # It commits whenever it HAS a target. A view in which nothing matches the
    # instruction is a block for this baseline too — counting it as an unsafe
    # commit would inflate the baseline's error rate in CoAct's favour.
    sa = score([t.a_xy is not None for t in trials], [t.a_correct for t in trials])
    sa["description"] = "One camera grounds the instruction and commits. No gate."

    # --- CoAct --------------------------------------------------------------
    co_committed = [not t.coact_blocked for t in trials]
    co_correct = [bool(t.coact_commit_correct) for t in trials]
    co = score(co_committed, co_correct)
    co["description"] = ("Two viewpoints must localise the referent to within "
                         f"{AGREE_TOL_M * 100:.0f} cm of the same point before anything moves.")
    coact_block_rate = co["block_pct"]

    # --- confidence gate, swept, then matched to CoAct's block rate --------
    sweep = []
    for tau in sorted({round(t.a_conf, 3) for t in trials}):
        committed = [t.a_xy is not None and t.a_conf >= tau for t in trials]
        row = score(committed, [t.a_correct for t in trials])
        row["tau"] = tau
        sweep.append(row)

    # An iso-availability comparison: give the single-agent baseline the same
    # licence to refuse that CoAct takes, and see what it does with it.
    matched = min(sweep, key=lambda r: abs((r["block_pct"] or 0) - (coact_block_rate or 0)))
    best_safety = min(sweep, key=lambda r: (r["unsafe_commit_pct"], -r["correct_commit_pct"]))
    cg = {
        "description": "One camera, blocking when its own confidence < tau.",
        "matched_to_coact_block_rate": matched,
        "best_achievable_safety": best_safety,
        "sweep": sweep,
    }

    # --- CoAct plus one spoken correction ----------------------------------
    # The fair end-to-end number. A block is not the end of the interaction:
    # the human says which plate they meant and the agents try again. What
    # matters is where the episode lands, not where it paused.
    ac_committed, ac_correct = [], []
    for t in trials:
        if not t.coact_blocked:
            ac_committed.append(True)
            ac_correct.append(bool(t.coact_commit_correct))
        else:
            ac_committed.append(bool(t.recovered))
            ac_correct.append(bool(t.recovery_correct))
    ac = score(ac_committed, ac_correct)
    ac["description"] = "CoAct, then one spoken correction on every block."

    # --- which disagreements a spoken correction can actually repair -------
    recovery_by_type: dict[str, dict] = {}
    for t in trials:
        if t.recovered is None:
            continue
        k = t.disagreement_type or "unknown"
        r = recovery_by_type.setdefault(k, {"blocks": 0, "recovered": 0, "on_target": 0})
        r["blocks"] += 1
        r["recovered"] += int(bool(t.recovered))
        r["on_target"] += int(bool(t.recovery_correct))
    for k, r in recovery_by_type.items():
        r["recovered_pct"] = round(100.0 * r["recovered"] / r["blocks"], 1)
        r["on_target_pct"] = round(100.0 * r["on_target"] / r["blocks"], 1)

    # --- by condition -------------------------------------------------------
    by_condition = {}
    for cond in sorted({t.condition for t in trials}):
        sub = [t for t in trials if t.condition == cond]
        d = len(sub)
        by_condition[cond] = {
            "trials": d,
            "single_agent_unsafe_pct": round(
                100.0 * sum(1 for t in sub if not t.a_correct) / d, 1),
            "coact_unsafe_pct": round(
                100.0 * sum(1 for t in sub
                            if not t.coact_blocked and not t.coact_commit_correct) / d, 1),
            "coact_block_pct": round(100.0 * sum(1 for t in sub if t.coact_blocked) / d, 1),
            "both_agents_agreed_and_were_wrong": sum(
                1 for t in sub if not t.coact_blocked and not t.coact_commit_correct),
        }

    # --- ambiguity detection, as a secondary view --------------------------
    amb = [t for t in trials if t.family == "ambiguous"]
    una_clean = [t for t in trials if t.family == "unambiguous" and t.condition == "clean"]
    tp = sum(1 for t in amb if t.coact_blocked)
    fp = sum(1 for t in una_clean if t.coact_blocked)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(amb) if amb else 0.0

    recov = [t for t in trials if t.recovered is not None]

    return {
        "meta": {
            "trials": n, "seed": seed, "elapsed_s": elapsed_s,
            "agree_tolerance_m": AGREE_TOL_M, "confidence_floor": CONFIDENCE_FLOOR,
            "ambiguous_instructions": len(amb),
            "unambiguous_instructions": n - len(amb),
            "conditions": sorted({t.condition for t in trials}),
        },
        "single_agent": sa,
        "confidence_gate": cg,
        "coact": co,
        "coact_after_correction": ac,
        "recovery_by_disagreement_type": recovery_by_type,
        "by_condition": by_condition,
        "ambiguity_detection": {
            "recall_pct": round(100.0 * rec, 1),
            "precision_vs_clean_unambiguous": round(prec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        },
        "correction": {
            "description": "After a block the human names the plate; both agents re-ground.",
            "attempts": len(recov),
            "consensus_restored_pct": (
                round(100.0 * sum(1 for t in recov if t.recovered) / len(recov), 1)
                if recov else None),
            "restored_on_intended_target_pct": (
                round(100.0 * sum(1 for t in recov if t.recovery_correct) / len(recov), 1)
                if recov else None),
        },
        "perception": {
            "mean_detections_per_view": round(
                sum(t.n_det_a + t.n_det_b for t in trials) / (2 * n), 2),
            "views_missing_a_plate_pct": round(
                100.0 * sum(1 for t in trials for k in (t.n_plate_a, t.n_plate_b) if k < 2)
                / (2 * n), 1),
        },
        "trials": [asdict(t) for t in trials],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the CoAct benchmark.")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="results/benchmark.json")
    ap.add_argument("--sizing", choices=["metric", "apparent", "both"], default="metric",
                    help="How agents judge object size. 'both' runs the ablation.")
    args = ap.parse_args()

    runs = ["metric", "apparent"] if args.sizing == "both" else [args.sizing]
    reports = {}
    for mode in runs:
        print(f"Running {args.trials} trials (seed {args.seed}, {mode} sizing)…", flush=True)
        reports[mode] = run(args.trials, args.seed, verbose=True,
                            metric_sizing=(mode == "metric"))
    report = reports[runs[0]]
    if len(runs) > 1:
        report["ablation_apparent_sizing"] = {
            k: v for k, v in reports["apparent"].items() if k != "trials"
        }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))

    m = report["meta"]
    print(f"\n{m['trials']} trials in {m['elapsed_s']}s over conditions "
          f"{', '.join(m['conditions'])}\n")
    hdr = f"  {'system':<34}{'correct':>9}{'unsafe':>9}{'blocked':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = [
        ("single agent (no gate)", report["single_agent"]),
        (f"confidence gate @ tau="
         f"{report['confidence_gate']['matched_to_coact_block_rate']['tau']}",
         report["confidence_gate"]["matched_to_coact_block_rate"]),
        ("CoAct consensus gate", report["coact"]),
        ("CoAct + one spoken correction", report["coact_after_correction"]),
    ]
    for name, r in rows:
        print(f"  {name:<34}{str(r['correct_commit_pct']) + '%':>9}"
              f"{str(r['unsafe_commit_pct']) + '%':>9}{str(r['block_pct']) + '%':>9}")

    print("\n  unsafe commits by condition")
    for cond, d in report["by_condition"].items():
        print(f"    {cond:<14} single agent {d['single_agent_unsafe_pct']:>5}%   "
              f"CoAct {d['coact_unsafe_pct']:>5}%   (CoAct blocked {d['coact_block_pct']}%)")

    print("\n  can a spoken correction repair this kind of disagreement?")
    for k, r in report["recovery_by_disagreement_type"].items():
        print(f"    {k:<14} {r['blocks']:>3} blocks   repaired on the intended "
              f"target {r['on_target_pct']}%")

    cr = report["correction"]
    print(f"\n  overall: consensus restored "
          f"{cr['consensus_restored_pct']}%, on the intended target "
          f"{cr['restored_on_intended_target_pct']}%")
    if "ablation_apparent_sizing" in report:
        ab = report["ablation_apparent_sizing"]
        print("\n  ablation — size judged from pixel area instead of measured radius")
        print(f"    unsafe commits (single agent) {ab['single_agent']['unsafe_commit_pct']}%"
              f"   CoAct block rate {ab['coact']['block_pct']}%"
              f"   CoAct unsafe {ab['coact']['unsafe_commit_pct']}%")
        for k, r in ab["recovery_by_disagreement_type"].items():
            print(f"    {k:<14} {r['blocks']:>3} blocks   repaired {r['on_target_pct']}%")

    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
