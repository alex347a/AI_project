"""
Dynamic weighted A* baseline (initial weight 2, decay 0.99) on every case.

The paper (v2, Table 1 caption) says only "dynamic weighted A* with initial weight of 2 and
0.99 decay". The decay rule below is the variant that reproduces the paper's numbers:

    h_weighted(s) = w * hypot(goal - s)
    w starts at 2 and is multiplied by 0.99 each time a node is expanded (never below 1).
    After every decay the whole OPEN queue is re-scored with the new weight, so all queued
    nodes always use the same current weight.

It subclasses the package's own AStar, so moves, costs, collision checks and counting
(operation = number of pops, storage = len(g)) are exactly the package's.

Needs table1_benchmark.py in the same folder (it reuses its helpers).

Usage (PowerShell, from the llm-astar folder):
    python dwa_baseline.py --max-cases 100        # quick test
    python dwa_baseline.py                        # all cases
"""
import argparse
import contextlib
import copy
import csv
import heapq
import io
import json
import math
import os

import table1_benchmark as tb
from llmastar.pather import AStar

W0, DECAY, FLOOR = 2.0, 0.99, 1.0     # initial weight, decay per expansion, minimum weight


class DynamicWAStar(AStar):
    def __init__(self):
        super().__init__()
        self.w = W0

    def get_neighbor(self, s):
        # Called once per expanded (popped) non-goal node.
        self.w = max(FLOOR, self.w * DECAY)                        # 1. decay the weight
        self.OPEN = [(self.f_value(n), n) for _, n in self.OPEN]   # 2. re-score the queue
        heapq.heapify(self.OPEN)
        return super().get_neighbor(s)                             # 3. neighbors pushed with new w

    def heuristic(self, s):
        return self.w * math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1])


def quiet(fn, *a, **k):
    """The package prints the query and result for every search; hide that."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/environment_50_30.json")
    p.add_argument("--outdir", default="results_dwa")
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()

    with open(args.dataset, "r") as f:
        environments = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "dwa_refresh_w2_d0.99.csv")

    n = errors = 0
    ors, srs, rls, valids = [], [], [], 0
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(tb.FIELDS)
        for env in environments:
            obstacles = tb.get_obstacles(env)
            bounds = (tuple(env["range_x"]), tuple(env["range_y"]))
            for sg in env["start_goal"]:
                if args.max_cases is not None and n >= args.max_cases:
                    break
                start, goal = list(sg[0]), list(sg[1])
                query = {"start": start, "goal": goal,
                         "size": [env["range_x"][1], env["range_y"][1]],
                         "horizontal_barriers": env["horizontal_barriers"],
                         "vertical_barriers": env["vertical_barriers"],
                         "range_x": env["range_x"], "range_y": env["range_y"]}
                n += 1
                status, err = "ok", ""
                a_ops = a_st = a_len = d_ops = d_st = d_len = None
                valid = False
                try:
                    res = quiet(AStar().searching, query=copy.deepcopy(query))
                    _, a_ops, a_st, a_len = tb.parse_result(res)
                    tb._last["path"] = None
                    res = quiet(DynamicWAStar().searching, query=copy.deepcopy(query))
                    _, d_ops, d_st, d_len = tb.parse_result(res)
                    path = tb._last["path"]            # package A* returns it goal -> start
                    valid = bool(path) and tb.path_is_valid(path[::-1], start, goal, obstacles, bounds)
                except Exception as e:
                    status, err = "error", f"{type(e).__name__}: {e}"
                    errors += 1

                orr = d_ops / a_ops if (status == "ok" and a_ops) else None
                sr = d_st / a_st if (status == "ok" and a_st) else None
                rl = d_len / a_len if (status == "ok" and a_len and d_len is not None) else None
                if orr is not None:
                    ors.append(orr)
                    srs.append(sr)
                    valids += valid
                    if valid and rl is not None:
                        rls.append(rl)
                w.writerow([env.get("id"), json.dumps(start), json.dumps(goal),
                            tb.fmt(a_ops), tb.fmt(a_st), tb.fmt(a_len),
                            tb.fmt(d_ops), tb.fmt(d_st), tb.fmt(d_len),
                            "" if orr is None else f"{orr:.4f}",
                            "" if sr is None else f"{sr:.4f}",
                            "" if rl is None else f"{rl:.4f}",
                            0, valid, status, err])
                if n % 100 == 0:
                    print(f"  {n} cases done")
            if args.max_cases is not None and n >= args.max_cases:
                break

    print(f"\nCases run: {n}   errors: {errors}")
    if ors:
        print(f"\nDynamic WA* (w0={W0:g}, decay={DECAY:g}, floor={FLOOR:g}, queue refreshed), "
              f"geometric means (%):")
        print(f"  OR  = {tb.geo_mean(ors) * 100:.2f}")
        print(f"  SR  = {tb.geo_mean(srs) * 100:.2f}")
        print(f"  RL  = {tb.geo_mean(rls) * 100:.2f}")
        print(f"  VPR = {valids / len(ors) * 100:.2f}")
        print("\nPaper, Dynamic WA*: OR 60.91, SR 78.53, RL 100.24, VPR 100")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()