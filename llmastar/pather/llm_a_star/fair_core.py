"""
fair_core.py -- apples-to-apples helpers for llmastar/pather/llm_a_star/ (next to optimal_core.py).

Everything runs on the SAME flat Grid (bytearray blocked map, array g/parent tables).

  grid_astar        plain A*, same Grid and same heuristic the optimal search uses
  grid_llm_astar    the ORIGINAL LLM-A* (h = d(goal) + d(current target), OPEN re-keyed)
  solve_counted     optimal UNIDIRECTIONAL LLM-A*: forward A* with an optional incumbent bound
                    (prunes f > incumbent cost) and an optional guide tie-break. Also counts the
                    work done OUTSIDE the main search: incumbent hop A* expansions and guide cells.
"""
import heapq
import math
from array import array

from .optimal_core import Grid, INF, build_guide  # noqa: F401 (Grid re-exported)

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
    heap = [(h(sx, sy, gx, gy), start)]
    expanded, touched, peak = 0, 1, 1
    while heap:
        _, i = _pop(heap)
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
                _push(heap, (ng + h(x + dx, y + dy, gx, gy), n))
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


# --------------------------------------------------------------------------- optimal unidirectional search
def _hop_counted(grid, a, b, max_expansions):
    W, blocked, moves, h = grid.W, grid.blocked, grid.moves, grid.h
    by, bx = divmod(b, W)
    ay, ax = divmod(a, W)
    g = {a: 0.0}
    par = {a: a}
    closed = set()
    heap = [(h(ax, ay, bx, by), a)]
    expanded = 0
    while heap:
        _, i = _pop(heap)
        if i in closed:
            continue
        if i == b:
            path = [b]
            while path[-1] != a:
                path.append(par[path[-1]])
            return path[::-1], expanded
        closed.add(i)
        expanded += 1
        if expanded > max_expansions:
            return None, expanded
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
                _push(heap, (ng + h(x + dx, y + dy, bx, by), n))
    return None, expanded


def _incumbent_counted(grid, start, goal, waypoints, cap_factor):
    """Stitch hop paths start -> waypoints -> goal. Returns ((path, cost) | None, hop expansions)."""
    total = 0
    path, cur, W = [start], start, grid.W
    for w in list(waypoints) + [goal]:
        if w == cur or grid.blocked[w]:
            continue
        (cy, cx), (wy, wx) = divmod(cur, W), divmod(w, W)
        cap = int(cap_factor * (grid.h(cx, cy, wx, wy) + 10))
        hop, n = _hop_counted(grid, cur, w, cap)
        total += n
        if hop is None:
            if w == goal:
                return None, total
            continue
        path.extend(hop[1:])
        cur = w
    if cur != goal:
        return None, total
    return (path, grid.path_cost(path)), total


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


def solve_counted(grid, start, goal, waypoints=(), use_incumbent=True, use_guide=True,
                  guide_radius=2, hop_cap_factor=20, guide_first=True):
    """
    stats["work"] = main-search expansions + incumbent hop expansions + guide-BFS cells.
    """
    incumbent, hop_exp = None, 0
    if use_incumbent and waypoints:
        incumbent, hop_exp = _incumbent_counted(grid, start, goal, waypoints, hop_cap_factor)

    guide, guide_cells = None, 0
    if use_guide and (incumbent is not None or waypoints):
        cells = incumbent[0] if incumbent is not None else [start, *waypoints, goal]
        guide = build_guide(grid, cells, guide_radius)
        guide_cells = len(guide) - guide.count(guide_radius + 1)   # cells the BFS reached

    path, cost, stats = unidirectional_search(grid, start, goal, guide, incumbent, guide_first)
    if path is None:
        return None
    stats["hop_expansions"] = hop_exp
    stats["guide_cells"] = guide_cells
    stats["work"] = stats["expanded"] + hop_exp + guide_cells
    stats["storage"] = stats["touched"] + (len(incumbent[0]) if incumbent else 0) + guide_cells
    return {"cost": cost, "path": path, "expanded": stats["expanded"], "work": stats["work"],
            "touched": stats["touched"], "storage": stats["storage"],
            "peak_open": stats["peak_open"], "stats": stats}
