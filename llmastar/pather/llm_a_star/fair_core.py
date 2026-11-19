"""
fair_core.py -- apples-to-apples helpers for llmastar/pather/llm_a_star/ (next to optimal_core.py).

Everything runs on the SAME flat Grid (bytearray blocked map, array g/parent tables).

  grid_astar         plain A*, same Grid, same heuristic and same tie-breaking as the optimal searches
  grid_llm_astar     the ORIGINAL LLM-A* (h = d(goal) + d(current target), OPEN re-keyed)
  solve_counted_uni  optimal UNIDIRECTIONAL LLM-A*: forward A* with an optional incumbent bound
                     (prunes f >= incumbent cost) and an optional guide tie-break
  solve_counted_bi   optimal BIDIRECTIONAL LLM-A* (optimal_core.bidirectional_search), same options

Tie-breaking: every optimal search here (grid_astar, the incumbent hops, unidirectional_search,
bidirectional_search) orders its heap by (round(f, 9), h, ...), so among equal-f nodes the one
closest to the target is expanded first. Without this, A* wanders through the plateau of nodes
whose f equals the optimal cost, which made plain A* look worse than it is.

Counting convention (identical for solve_counted_uni and solve_counted_bi):
  ops      = main-search expansions + incumbent hop expansions
  work     = ops + guide-BFS cells
  storage  = main-search touched nodes + incumbent hop touched nodes (upper bound: a node seen
             by both the hops and the search counts twice)
"""
import heapq
import math
from array import array

from .optimal_core import Grid, INF, bidirectional_search, build_guide  # noqa: F401 (Grid re-exported)

_push, _pop = heapq.heappush, heapq.heappop


def _rebuild(par, i):
    path = [i]
    while par[path[-1]] != path[-1]:
        path.append(par[path[-1]])
    return path[::-1]


# --------------------------------------------------------------------------- plain A*
def grid_astar(grid, start, goal):
    W = grid.W
    N = W * grid.H
    blocked, moves, h = grid.blocked, grid.moves, grid.h
    gy, gx = divmod(goal, W)
    sy, sx = divmod(start, W)
    g = array("d", [INF]) * N
    par = array("i", [-1]) * N
    closed = bytearray(N)
    g[start] = 0.0
    par[start] = start
    h0 = h(sx, sy, gx, gy)
    heap = [(round(h0, 9), h0, start)]
    expanded, touched, peak = 0, 1, 1
    while heap:
        _, _, i = _pop(heap)
        if closed[i]:
            continue
        closed[i] = 1
        expanded += 1
        if i == goal:
            break
        y, x = divmod(i, W)
        gi = g[i]
        for dx, dy, off, c in moves:
            n = i + off
            if blocked[n] or closed[n]:
                continue
            ng = gi + c
            if ng < g[n]:
                if g[n] == INF:
                    touched += 1
                g[n] = ng
                par[n] = i
                hn = h(x + dx, y + dy, gx, gy)
                _push(heap, (round(ng + hn, 9), hn, n))
        peak = max(peak, len(heap))
    if g[goal] == INF:
        return None
    path = _rebuild(par, goal)
    return {"cost": grid.path_cost(path), "path": path, "expanded": expanded,
            "work": expanded, "touched": touched, "storage": touched, "peak_open": peak}


# --------------------------------------------------------------------------- original LLM-A*
def grid_llm_astar(grid, start, goal, target_list):
    """
    Original LLM-A* semantics. target_list = [start, *waypoints, goal] as grid indices.
    Euclidean h(s) = d(s, goal) + d(s, target). Stale heap entries are skipped.
    (Deliberately NOT given the tie-break change: it reproduces the original algorithm.)
    """
    W = grid.W
    N = W * grid.H
    blocked, moves = grid.blocked, grid.moves
    hypot = math.hypot
    gy, gx = divmod(goal, W)
    sy, sx = divmod(start, W)
    coords = [divmod(t, W) for t in target_list]          # (y, x)
    ti = 1 if len(target_list) > 1 else 0
    tidx = target_list[ti]
    ty, tx = coords[ti]
    g = array("d", [INF]) * N
    par = array("i", [-1]) * N
    closed = bytearray(N)
    g[start] = 0.0
    par[start] = start
    heap = [(hypot(gx - sx, gy - sy) + hypot(tx - sx, ty - sy), sx, sy)]
    expanded, touched, peak = 0, 1, 1
    while heap:
        _, x, y = _pop(heap)
        i = y * W + x
        if closed[i]:
            continue
        closed[i] = 1
        expanded += 1
        if i == goal:
            break
        gi = g[i]
        for dx, dy, off, c in moves:
            n = i + off
            if n == tidx and goal != tidx:
                ti += 1
                if ti < len(target_list):
                    tidx = target_list[ti]
                    ty, tx = coords[ti]
                heap = [(g[yy * W + xx] + hypot(gx - xx, gy - yy) + hypot(tx - xx, ty - yy), xx, yy)
                        for _, xx, yy in heap]
                heapq.heapify(heap)
            if blocked[n] or closed[n]:
                continue
            ng = gi + c
            if ng < g[n]:
                if g[n] == INF:
                    touched += 1
                g[n] = ng
                par[n] = i
                nx, ny = x + dx, y + dy
                _push(heap, (ng + hypot(gx - nx, gy - ny) + hypot(tx - nx, ty - ny), nx, ny))
        peak = max(peak, len(heap))
    if g[goal] == INF:
        return None
    path = _rebuild(par, goal)
    return {"cost": grid.path_cost(path), "path": path, "expanded": expanded,
            "work": expanded, "touched": touched, "storage": touched, "peak_open": peak}


# --------------------------------------------------------------------------- incumbent hops
def _hop_counted(grid, a, b, max_expansions):
    """Unidirectional A* hop. Returns (path | None, expanded, touched)."""
    W, blocked, moves, h = grid.W, grid.blocked, grid.moves, grid.h
    by, bx = divmod(b, W)
    ay, ax = divmod(a, W)
    g = {a: 0.0}
    par = {a: a}
    closed = set()
    h0 = h(ax, ay, bx, by)
    heap = [(round(h0, 9), h0, a)]
    expanded = 0
    while heap:
        _, _, i = _pop(heap)
        if i in closed:
            continue
        if i == b:
            path = [b]
            while path[-1] != a:
                path.append(par[path[-1]])
            return path[::-1], expanded, len(g)
        closed.add(i)
        expanded += 1
        if expanded > max_expansions:
            return None, expanded, len(g)
        y, x = divmod(i, W)
        gi = g[i]
        for dx, dy, off, c in moves:
            n = i + off
            if blocked[n] or n in closed:
                continue
            ng = gi + c
            if ng < g.get(n, INF):
                g[n] = ng
                par[n] = i
                hn = h(x + dx, y + dy, bx, by)
                _push(heap, (round(ng + hn, 9), hn, n))
    return None, expanded, len(g)


def _incumbent_counted(grid, start, goal, waypoints, cap_factor):
    """Stitch hop paths start -> waypoints -> goal.

    Returns ((path, cost) | None, total hop expansions, total hop touched nodes).
    """
    total, total_t = 0, 0
    path, cur, W = [start], start, grid.W
    for w in list(waypoints) + [goal]:
        if w == cur or grid.blocked[w]:
            continue
        (cy, cx), (wy, wx) = divmod(cur, W), divmod(w, W)
        cap = int(cap_factor * (grid.h(cx, cy, wx, wy) + 10))
        hop, n, t = _hop_counted(grid, cur, w, cap)
        total += n
        total_t += t
        if hop is None:
            if w == goal:
                return None, total, total_t
            continue
        path.extend(hop[1:])
        cur = w
    if cur != goal:
        return None, total, total_t
    return (path, grid.path_cost(path)), total, total_t


# --------------------------------------------------------------------------- optimal unidirectional search
EPS = 1e-9


def unidirectional_search(grid, start, goal, guide=None, incumbent=None, guide_first=True):
    """
    Forward A*. f = g + h is admissible and consistent, so the path is optimal.
    incumbent = (path, cost): children with f >= cost - EPS are never pushed (same rule as
    bidirectional_search). If nothing better is found, the incumbent itself is the answer.
    guide: per-cell table, smaller = closer to the LLM plan; breaks ties between equal-f nodes.
    """
    W = grid.W
    N = W * grid.H
    blocked, moves, h = grid.blocked, grid.moves, grid.h
    gy, gx = divmod(goal, W)
    sy, sx = divmod(start, W)
    ub = incumbent[1] - EPS if incumbent is not None else INF
    g = array("d", [INF]) * N
    par = array("i", [-1]) * N
    closed = bytearray(N)
    g[start] = 0.0
    par[start] = start
    h0 = h(sx, sy, gx, gy)
    d0 = guide[start] if guide is not None else 0
    f0 = round(h0, 9)
    heap = [(f0, d0, h0, start) if guide_first else (f0, h0, d0, start)]
    expanded, touched, peak = 0, 1, 1
    while heap:
        _, _, _, i = _pop(heap)
        if closed[i]:
            continue
        closed[i] = 1
        expanded += 1
        if i == goal:
            break
        y, x = divmod(i, W)
        gi = g[i]
        for dx, dy, off, c in moves:
            n = i + off
            if blocked[n] or closed[n]:
                continue
            ng = gi + c
            if ng < g[n]:
                hn = h(x + dx, y + dy, gx, gy)
                f = round(ng + hn, 9)
                if f >= ub:
                    continue            # cannot beat the incumbent (h is admissible)
                if g[n] == INF:
                    touched += 1
                g[n] = ng
                par[n] = i
                d = guide[n] if guide is not None else 0
                _push(heap, (f, d, hn, n) if guide_first else (f, hn, d, n))
        peak = max(peak, len(heap))
    stats = {"expanded": expanded, "touched": touched, "peak_open": peak}
    if g[goal] == INF:
        if incumbent is not None:       # nothing beat the incumbent, so it is optimal
            return incumbent[0], incumbent[1], stats
        return None, None, stats
    path = _rebuild(par, goal)
    return path, grid.path_cost(path), stats


def solve_counted_uni(grid, start, goal, waypoints=(), use_incumbent=True, use_guide=True,
                      guide_radius=2, hop_cap_factor=20, guide_first=True):
    """Optimal UNIDIRECTIONAL search, counted as described in the module docstring."""
    incumbent, hop_exp, hop_touched = None, 0, 0
    if use_incumbent and waypoints:
        incumbent, hop_exp, hop_touched = _incumbent_counted(grid, start, goal, waypoints, hop_cap_factor)

    guide, guide_cells = None, 0
    if use_guide and (incumbent is not None or waypoints):
        cells = incumbent[0] if incumbent is not None else [start, *waypoints, goal]
        guide = build_guide(grid, cells, guide_radius)
        guide_cells = len(guide) - guide.count(guide_radius + 1)   # cells the BFS reached

    path, cost, stats = unidirectional_search(grid, start, goal, guide, incumbent, guide_first)
    if path is None:
        return None
    stats["hop_expansions"] = hop_exp
    stats["hop_touched"] = hop_touched
    stats["guide_cells"] = guide_cells
    stats["ops"] = stats["expanded"] + hop_exp
    stats["work"] = stats["ops"] + guide_cells
    stats["storage"] = stats["touched"] + hop_touched      # upper bound: overlap counts twice
    return {"cost": cost, "path": path, "expanded": stats["expanded"], "ops": stats["ops"],
            "work": stats["work"], "touched": stats["touched"], "storage": stats["storage"],
            "peak_open": stats["peak_open"], "stats": stats}


# --------------------------------------------------------------------------- optimal bidirectional search
def solve_counted_bi(grid, start, goal, waypoints=(), use_incumbent=True, use_guide=True,
                     guide_radius=2, hop_cap_factor=20, guide_first=True):
    """Optimal BIDIRECTIONAL search with the same counting as solve_counted_uni."""
    incumbent, hop_exp, hop_touched = None, 0, 0
    if use_incumbent and waypoints:
        incumbent, hop_exp, hop_touched = _incumbent_counted(grid, start, goal, waypoints, hop_cap_factor)

    guide, guide_cells = None, 0
    if use_guide and (incumbent is not None or waypoints):
        cells = incumbent[0] if incumbent is not None else [start, *waypoints, goal]
        guide = build_guide(grid, cells, guide_radius)
        guide_cells = len(guide) - guide.count(guide_radius + 1)

    path, cost, stats, _ = bidirectional_search(grid, start, goal, guide, incumbent, False, guide_first)
    if path is None:
        return None
    stats["hop_expansions"] = hop_exp
    stats["hop_touched"] = hop_touched
    stats["guide_cells"] = guide_cells
    stats["ops"] = stats["expanded"] + hop_exp
    stats["work"] = stats["ops"] + guide_cells
    stats["storage"] = stats["touched"] + hop_touched
    return {"cost": cost, "path": path, "expanded": stats["expanded"], "ops": stats["ops"],
            "work": stats["work"], "touched": stats["touched"], "storage": stats["storage"],
            "stats": stats}