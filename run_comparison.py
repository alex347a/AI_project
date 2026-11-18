#!/usr/bin/env python3
"""
run_comparison.py -- original LLM-A* vs. optimal unidirectional LLM-A* on the whole dataset.

Put this file in the repo root (next to setup.py) and run it from there:

    # .env file in the repo root (or the folder you run from):
    #     GEMINI_API_KEY=...        (GOOGLE_API_KEY also works)
    #     OPENAI_API_KEY=...        (only if you want to use OpenAI instead)
    #     GEMINI_MODEL=gemini-3.5-flash     (optional)

    python run_comparison.py                          # all 100 maps x 10 start/goal pairs
    python run_comparison.py --maps 0-4 --pairs 3     # quick subset
    python run_comparison.py --llm gemini             # force a provider (default: whichever key is in .env)
    python run_comparison.py --llm mock               # no key: straight-line "LLM", pipeline test only

What it does
------------
For every (map, start/goal pair) in dataset/environment_50_30.json it runs

    A*                  repo's AStar               -> reference for the ratios
    original LLM-A*     repo's LLMAStar            (untouched)
    new LLM-A*          UniLLMAStarFast            (uni_llm_a.py + optimal_core.py)

Both LLM-A* variants get the SAME LLM answer: it is requested once per prompt and cached
(llm_cache_<provider>_<model>.json), so the comparison is not affected by LLM randomness and
re-runs cost nothing. Keys are read from .env (a file, or a folder of files containing KEY=VALUE
lines); variables already set in the shell win over .env.

Metrics (per query, then averaged)
----------------------------------
    OR   operation ratio  = method "operation" / A* "operation"   (lower is better)
    SR   storage ratio    = method "storage"   / A* "storage"     (lower is better)
    RL   relative length  = method path length / A* path length   (1.0 = as short as A*)
    VPR  valid path ratio = share of queries whose returned path is valid (higher is better)

OR / SR / RL are averaged over the queries where both A* and the method returned a valid path.
A path is valid if it starts at the start, ends at the goal, every step is one of the 8 unit
moves and no visited cell lies on a barrier or on the border. A search that crashes or finds no
path counts as invalid.

"operation" and "storage" are whatever each class reports:
    A*, original   : operation = pops from OPEN (stale duplicates included), storage = len(g)
    new            : operation = expanded nodes, storage = touched nodes + incumbent path + guide cells
The per-query CSV also has new_OR_work, which uses "work" (expansions + incumbent hop
expansions + guide cells) instead of expansions only.

Nothing in the repo is modified: plotting is switched off and the new algorithm's path is
captured from outside, only inside this script.

Outputs (in --out, default results/)
    per_query.csv   one row per query (resumable: finished queries are skipped on re-run)
    per_map.csv     OR / SR / RL / VPR per map
    summary.json    overall numbers
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")          # never open a plot window

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from llmastar.env.search import plotting                                    # noqa: E402
from llmastar.pather import AStar, LLMAStar                                 # noqa: E402
from llmastar.pather.llm_a_star import uni_llm_a as uni_mod                 # noqa: E402
from llmastar.pather.llm_a_star.uni_llm_a import UniLLMAStarFast            # noqa: E402
from llmastar.pather.llm_a_star.optimal_core import Grid                    # noqa: E402

plotting.Plotting.animation = lambda *a, **k: None   # the original plots (and blocks) on every call

MOTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # Env.motions


# --------------------------------------------------------------------------- .env
def _parse_env_file(path):
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] in ("'", '"') and value.find(value[0], 1) != -1:
                value = value[1:value.find(value[0], 1)]          # quoted; anything after is a comment
            else:
                value = value.split(" #", 1)[0].strip()
            if key:
                os.environ.setdefault(key, value)


def load_env(*dirs):
    """Load KEY=VALUE pairs from <dir>/.env (a file, or a folder holding files). Shell variables win."""
    seen = set()
    for d in dirs:
        p = os.path.abspath(os.path.join(d, ".env"))
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            _parse_env_file(p)
        elif os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp):
                    _parse_env_file(fp)


def gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def pick_provider(requested):
    if requested != "auto":
        return requested
    g, o = gemini_key(), os.environ.get("OPENAI_API_KEY")
    if g and not o:
        return "gemini"
    if o and not g:
        return "gpt"
    if g and o:
        sys.exit("Both a Gemini and an OpenAI key were found; choose with --llm gemini or --llm gpt.")
    sys.exit("No API key found. Put GEMINI_API_KEY=... (or OPENAI_API_KEY=...) in a .env file "
             "in the repo root, or use --llm mock to test without an LLM.")


# --------------------------------------------------------------------------- LLM access
class LLMUnavailable(RuntimeError):
    pass


class LLMConfigError(LLMUnavailable):
    """Bad key / model name / request: retrying will not help, so the whole run stops."""


class GeminiBackend:
    """Minimal Gemini client over the public REST API (no extra package needed)."""
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

    def __init__(self, api_key, model, timeout=60, thinking="low"):
        self.api_key, self.model, self.timeout, self.thinking = api_key, model, timeout, thinking

    def _generation_config(self):
        cfg = {"maxOutputTokens": 8192}                      # room for thinking tokens + the path
        if self.thinking != "default":
            if self.model.startswith("gemini-3"):            # Gemini 3.x: thinkingLevel (3.5 Flash defaults to medium)
                cfg["thinkingConfig"] = {"thinkingLevel": self.thinking}
            elif self.model.startswith("gemini-2.5-flash") and self.thinking == "minimal":
                cfg["thinkingConfig"] = {"thinkingBudget": 0}  # Gemini 2.5 Flash: budget 0 switches thinking off
        return cfg

    def ask(self, prompt=None, *args, stop=None, max_tokens=None, **kwargs):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": self._generation_config()}
        req = urllib.request.Request(
            self.URL.format(self.model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300].replace(self.api_key, "***")
            if e.code in (400, 401, 403, 404):
                raise LLMConfigError(f"Gemini API HTTP {e.code} (model {self.model!r}): {detail}")
            raise RuntimeError(f"Gemini API HTTP {e.code}: {detail}")
        cands = data.get("candidates") or []
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ValueError("Gemini returned an empty response (blocked or cut off)")
        return text


class LLMProxy:
    """Stands in for `.model`: caches answers by prompt so both algorithms see the same plan."""

    def __init__(self, backend, cache_path=None, mock=False, retries=3, delay=0.0):
        self.backend, self.mock, self.retries, self.delay = backend, mock, retries, delay
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)
        self.spent = 0.0            # seconds spent in real LLM calls since reset()
        self.current_query = None   # only used by the mock

    def reset(self):
        self.spent = 0.0

    def _save(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _mock_answer(q, n=3):
        (sx, sy), (gx, gy) = q["start"], q["goal"]
        pts = [[sx, sy]]
        pts += [[round(sx + (gx - sx) * k / (n + 1)), round(sy + (gy - sy) * k / (n + 1))]
                for k in range(1, n + 1)]
        pts.append([gx, gy])
        return f"Generated Path: {pts}"

    def ask(self, *args, **kwargs):
        prompt = kwargs.get("prompt", args[0] if args else None)
        if self.mock:
            return self._mock_answer(self.current_query)
        if prompt in self.cache:
            return self.cache[prompt]
        last = None
        if self.delay:
            time.sleep(self.delay)
        for attempt in range(self.retries):
            t = time.perf_counter()
            try:
                print("  asking the LLM ...", file=sys.stderr, flush=True)
                answer = self.backend.ask(*args, **kwargs)
                took = time.perf_counter() - t
                self.spent += took
                print(f"  LLM answered in {took:.1f}s", file=sys.stderr, flush=True)
                self.cache[prompt] = answer
                self._save()
                return answer
            except LLMConfigError:
                raise
            except Exception as e:          # network / rate limit / empty answer
                self.spent += time.perf_counter() - t
                last = e
                wait = 2 ** (attempt + 1)
                print(f"  LLM call failed after {time.perf_counter() - t:.1f}s "
                      f"(attempt {attempt + 1}/{self.retries}): {e!r}"
                      + (f"; retrying in {wait}s" if attempt < self.retries - 1 else ""),
                      file=sys.stderr, flush=True)
                if attempt < self.retries - 1:
                    time.sleep(wait)
        raise LLMUnavailable(f"LLM call failed after {self.retries} attempts: {last!r}")


# --------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def quiet(verbose):
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def make_query(env, start, goal):
    return {"start": list(start), "goal": list(goal),
            "size": [env["range_x"][1], env["range_y"][1]],
            "horizontal_barriers": env["horizontal_barriers"],
            "vertical_barriers": env["vertical_barriers"],
            "range_x": list(env["range_x"]), "range_y": list(env["range_y"])}


def checker_grid(q):
    """Same blocked-cell model the search classes use (ranges are decremented by one)."""
    rx = [q["range_x"][0], q["range_x"][1] - 1]
    ry = [q["range_y"][0], q["range_y"][1] - 1]
    return Grid(rx, ry, q["horizontal_barriers"], q["vertical_barriers"], MOTIONS)


def is_valid(grid, path, start, goal):
    if not path or tuple(path[0]) != tuple(start) or tuple(path[-1]) != tuple(goal):
        return False
    moves = set(MOTIONS)
    try:
        for (x, y), (x2, y2) in zip(path, path[1:]):
            if (x2 - x, y2 - y) not in moves:
                return False
        return all(not grid.blocked[grid.index(x, y)] for x, y in path)
    except ValueError:                      # a cell outside the map
        return False


def parse_ranges(spec, upper):
    if spec in (None, "", "all"):
        return list(range(upper))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < upper]


# --------------------------------------------------------------------------- one query
def run_one(q, astar, orig, new, proxy, captured, verbose):
    grid = checker_grid(q)
    start, goal = q["start"], q["goal"]
    proxy.current_query = q
    row = {}

    # --- A* (reference)
    t = time.perf_counter()
    with quiet(verbose):
        ra = astar.searching(copy.deepcopy(q), filepath="temp.png")
    row.update(astar_ops=ra["operation"], astar_storage=ra["storage"],
               astar_len=ra["length"], astar_sec=time.perf_counter() - t)
    a_path = astar.extract_path(astar.PARENT)
    if a_path and tuple(a_path[0]) != tuple(start):          # AStar.extract_path returns goal -> start
        a_path = a_path[::-1]
    a_ok = is_valid(grid, a_path, start, goal)
    row["astar_valid"] = int(a_ok)

    # --- original LLM-A*
    proxy.reset()
    t = time.perf_counter()
    r, path, err = None, None, ""
    try:
        with quiet(verbose):
            r = orig.searching(copy.deepcopy(q), filepath="temp.png")
        path = orig.extract_path(orig.PARENT)
    except LLMUnavailable:
        raise
    except Exception as e:
        r, path, err = None, None, repr(e)
    sec = time.perf_counter() - t - proxy.spent
    fill(row, "orig", r, path, err, sec, grid, start, goal, a_ok, ra)
    row["n_waypoints"] = max(len(r["llm_output"]) - 2, 0) if r else ""

    # --- new unidirectional LLM-A*
    proxy.reset()
    captured.clear()
    t = time.perf_counter()
    r, path, err = None, None, ""
    try:
        with quiet(verbose):
            r = new.searching(copy.deepcopy(q), filepath="temp.png", show=False, plot=False)
        g, res = captured["grid"], captured["res"]
        path = [g.xy(i) for i in res["path"]]
        row["new_work"] = res["work"]
    except LLMUnavailable:
        raise
    except Exception as e:
        r, path, err = None, None, repr(e)
    sec = time.perf_counter() - t - proxy.spent
    fill(row, "new", r, path, err, sec, grid, start, goal, a_ok, ra)
    if r and a_ok:
        row["new_OR_work"] = row["new_work"] / ra["operation"]
    return row


def fill(row, p, r, path, err, sec, grid, start, goal, a_ok, ra):
    valid = bool(r) and is_valid(grid, path, start, goal)
    row[f"{p}_valid"] = int(valid)
    row[f"{p}_sec"] = sec
    row[f"{p}_error"] = err
    if r:
        row[f"{p}_ops"], row[f"{p}_storage"], row[f"{p}_len"] = r["operation"], r["storage"], r["length"]
    if valid and a_ok:
        row[f"{p}_OR"] = r["operation"] / ra["operation"]
        row[f"{p}_SR"] = r["storage"] / ra["storage"]
        row[f"{p}_RL"] = r["length"] / ra["length"]


COLUMNS = (["map_id", "pair", "sx", "sy", "gx", "gy", "n_waypoints",
            "astar_valid", "astar_ops", "astar_storage", "astar_len", "astar_sec"]
           + [f"{p}_{k}" for p in ("orig", "new")
              for k in ("valid", "ops", "storage", "len", "OR", "SR", "RL", "sec", "error")]
           + ["new_work", "new_OR_work"])


# --------------------------------------------------------------------------- aggregation
def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(rows, p):
    n = len(rows)
    ok = [r for r in rows if r.get(f"{p}_valid") == "1"]
    return {"queries": n, "valid": len(ok),
            "OR": _mean(ok, f"{p}_OR"), "SR": _mean(ok, f"{p}_SR"), "RL": _mean(ok, f"{p}_RL"),
            "VPR": len(ok) / n if n else float("nan"),
            "sec": _mean(rows, f"{p}_sec")}


def write_reports(out_dir):
    with open(os.path.join(out_dir, "per_query.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no results yet")
        return
    by_map = {}
    for r in rows:
        by_map.setdefault(int(r["map_id"]), []).append(r)

    with open(os.path.join(out_dir, "per_map.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map_id", "queries"] + [f"{p}_{k}" for p in ("orig", "new") for k in ("OR", "SR", "RL", "VPR")])
        for m in sorted(by_map):
            a = {p: aggregate(by_map[m], p) for p in ("orig", "new")}
            w.writerow([m, len(by_map[m])] + [round(a[p][k], 6) for p in ("orig", "new")
                                              for k in ("OR", "SR", "RL", "VPR")])

    overall = {"original": aggregate(rows, "orig"), "new": aggregate(rows, "new"),
               "new_OR_incl_overhead": _mean([r for r in rows if r.get("new_valid") == "1"], "new_OR_work")}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(f"\n{len(rows)} queries over {len(by_map)} maps")
    print(f"{'':<14}{'OR':>8}{'SR':>8}{'RL':>8}{'VPR':>8}{'sec/query':>11}")
    for name, key in (("original", "original"), ("new", "new")):
        a = overall[key]
        print(f"{name:<14}{a['OR']:>8.3f}{a['SR']:>8.3f}{a['RL']:>8.3f}{a['VPR']*100:>7.1f}%{a['sec']:>11.3f}")
    print(f"new OR if hop + guide work is counted as operations: {overall['new_OR_incl_overhead']:.3f}")
    print(f"files in {out_dir}/: per_query.csv, per_map.csv, summary.json")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "environment_50_30.json"))
    ap.add_argument("--llm", choices=["auto", "gemini", "gpt", "mock"], default="auto",
                    help="auto = whichever key is found in .env (default)")
    ap.add_argument("--gemini-model", default=None,
                    help="default: GEMINI_MODEL from .env, else gemini-3.5-flash")
    ap.add_argument("--gemini-thinking", choices=["minimal", "low", "medium", "high", "default"], default="low",
                    help="Gemini thinking effort (default: low; 'default' = model default, slow on 3.5 Flash)")
    ap.add_argument("--gemini-timeout", type=float, default=60.0, help="seconds per Gemini request (default 60)")
    ap.add_argument("--skip-check", action="store_true", help="skip the one-call connection check")
    ap.add_argument("--llm-delay", type=float, default=0.0,
                    help="seconds to wait before each real LLM call (for rate limits)")
    ap.add_argument("--prompt", choices=["standard", "cot", "repe"], default="standard")
    ap.add_argument("--maps", default="all", help="e.g. 0-4,7 (default: all)")
    ap.add_argument("--pairs", type=int, default=None, help="only the first N start/goal pairs per map")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default=None, help="LLM answer cache file (default: llm_cache_<provider>_<model>.json)")
    ap.add_argument("--fresh", action="store_true", help="ignore and overwrite existing results")
    ap.add_argument("--verbose", action="store_true", help="show the algorithms' own prints")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    map_ids = parse_ranges(args.maps, len(dataset))

    load_env(ROOT, os.getcwd())
    provider = pick_provider(args.llm)
    model_name = ""
    if provider == "gpt":
        import openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY not found (shell or .env).")
        openai.api_key = key
        model_name = "gpt-3.5-turbo-instruct"
    elif provider == "gemini":
        key = gemini_key()
        if not key:
            sys.exit("GEMINI_API_KEY (or GOOGLE_API_KEY) not found in the shell or in .env.")
        model_name = args.gemini_model or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
    print(f"LLM: {provider}" + (f" ({model_name})" if model_name else " (mock: straight-line waypoints)"), flush=True)

    astar = AStar()
    orig = LLMAStar(llm="gpt", prompt=args.prompt)          # builds the repo's own wrappers; .model is replaced below
    new = UniLLMAStarFast(llm="gpt", prompt=args.prompt)
    if provider == "gemini":
        backend = GeminiBackend(key, model_name, timeout=args.gemini_timeout, thinking=args.gemini_thinking)
        if not args.skip_check:
            print(f"Checking Gemini connection (thinking={args.gemini_thinking}, timeout {args.gemini_timeout:.0f}s) ...",
                  flush=True)
            t = time.perf_counter()
            try:
                reply = backend.ask(prompt="Reply with the single word OK.")
            except Exception as e:
                sys.exit(f"Gemini check failed after {time.perf_counter() - t:.1f}s: {e!r}\n"
                         "If this is a timeout: try --gemini-thinking minimal, another --gemini-model "
                         "(e.g. gemini-2.5-flash), or check VPN / proxy / firewall.")
            print(f"Gemini reachable: answered {reply.strip()[:20]!r} in {time.perf_counter() - t:.1f}s", flush=True)
    else:
        backend = orig.model                                 # the repo's ChatGPT wrapper (unused for mock)
    cache_path = None
    if provider != "mock":
        tag = model_name + (f"_think-{args.gemini_thinking}" if provider == "gemini" else "")
        safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in tag)
        cache_path = args.cache or f"llm_cache_{provider}_{safe}.json"
    proxy = LLMProxy(backend, cache_path=cache_path, mock=provider == "mock", delay=args.llm_delay)
    orig.model = new.model = proxy

    captured = {}
    real_solve = uni_mod.solve_counted

    def capturing_solve(grid, *a, **k):                      # grab the path without touching the repo
        res = real_solve(grid, *a, **k)
        captured["grid"], captured["res"] = grid, res
        return res
    uni_mod.solve_counted = capturing_solve

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "per_query.csv")
    done = set()
    if os.path.exists(csv_path) and not args.fresh:
        with open(csv_path, newline="", encoding="utf-8") as f:
            done = {(int(r["map_id"]), int(r["pair"])) for r in csv.DictReader(f)}
        mode = "a"
    else:
        mode = "w"

    total = sum(min(len(dataset[m]["start_goal"]), args.pairs or 10 ** 9) for m in map_ids)
    n = 0
    skipped_llm = 0
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if mode == "w":
            w.writeheader()
        for m in map_ids:
            env = dataset[m]
            for pair, (start, goal, _text) in enumerate(env["start_goal"][:args.pairs]):
                n += 1
                if (m, pair) in done:
                    continue
                q = make_query(env, start, goal)
                try:
                    row = run_one(q, astar, orig, new, proxy, captured, args.verbose)
                except LLMConfigError as e:
                    sys.exit(f"\nStopping: {e}\n(finished queries are saved; fix the key/model and re-run to resume)")
                except LLMUnavailable as e:
                    skipped_llm += 1
                    print(f"[{n}/{total}] map {m} pair {pair}: SKIPPED ({e})")
                    continue
                row.update(map_id=m, pair=pair, sx=start[0], sy=start[1], gx=goal[0], gy=goal[1])
                w.writerow({c: row.get(c, "") for c in COLUMNS})
                f.flush()

                def short(p):
                    return (f"OR={row[p + '_OR']:.2f} RL={row[p + '_RL']:.2f}" if p + "_OR" in row
                            else f"INVALID {row[p + '_error'][:40]}")
                print(f"[{n}/{total}] map {m} pair {pair}: A* ops={row['astar_ops']} | "
                      f"orig {short('orig')} | new {short('new')}")

    if skipped_llm:
        print(f"\n{skipped_llm} queries skipped because the LLM was unreachable; re-run to retry them.")
    write_reports(args.out)


if __name__ == "__main__":
    main()