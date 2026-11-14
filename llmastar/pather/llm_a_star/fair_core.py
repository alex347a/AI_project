"""
fair_core.py -- apples-to-apples helpers for llmastar/pather/llm_a_star/ (next to optimal_core.py).

Everything here runs on the SAME flat Grid (bytearray blocked map, array g/parent tables), so
speed and memory differences come from the algorithm, not from shapely / tuples / list-CLOSED.

  grid_astar        plain A*, same Grid and same (octile) heuristic Bi uses
  grid_llm_astar    the ORIGINAL LLM-A* (h = d(goal) + d(current target), target advances when a
                    neighbour equals it, OPEN is re-keyed), re-implemented on the Grid
  solve_counted     Bi LLM-A* that also counts the work done OUTSIDE the main search:
                    incumbent hop A* expansions and guide-BFS cells
"""
import heapq
import math
from array import array

from .optimal_core import Grid, INF, bidirectional_search, build_guide  # noqa: F401 (Grid re-exported)

_push, _pop = heapq.heappush, heapq.heappop


def _rebuild(grid, par, i, g):
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
    expanded, touched = 0, 1
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
    if g[goal] == INF:
        return None
    path = _rebuild(grid, par, goal, g)
    return {"cost": grid.path_cost(path), "path": path, "expanded": expanded,
            "work": expanded, "touched": touched}


# --------------------------------------------------------------------------- original LLM-A*
def grid_llm_astar(grid, start, goal, target_list):
    """
    Original LLM-A* semantics. target_list = [start, *waypoints, goal] as grid indices, exactly the
    list LLMAStar builds. Euclidean h(s) = d(s, goal) + d(s, target), like the original.
    Stale heap entries are skipped instead of re-expanded (same unique-closed count).
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
    expanded, touched = 0, 1
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
    if g[goal] == INF:
        return None
    path = _rebuild(grid, par, goal, g)
    return {"cost": grid.path_cost(path), "path": path, "expanded": expanded,
            "work": expanded, "touched": touched}


# --------------------------------------------------------------------------- Bi LLM-A* with full accounting
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
    """Same stitching as optimal_core.build_incumbent, but also returns total hop expansions."""
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


def solve_counted(grid, start, goal, waypoints=(), use_incumbent=True, use_guide=True,
                  guide_radius=2, hop_cap_factor=20, guide_first=True):
    """
    Same pipeline as optimal_core.solve, but stats["work"] = main-search expansions
    + incumbent hop expansions + guide-BFS cells, so nothing is hidden.
    """
    incumbent, hop_exp = None, 0
    if use_incumbent and waypoints:
        incumbent, hop_exp = _incumbent_counted(grid, start, goal, waypoints, hop_cap_factor)

    guide, guide_cells = None, 0
    if use_guide and (incumbent is not None or waypoints):
        cells = incumbent[0] if incumbent is not None else [start, *waypoints, goal]
        guide = build_guide(grid, cells, guide_radius)
        guide_cells = len(guide) - guide.count(guide_radius + 1)   # cells the BFS reached

    path, cost, stats, _ = bidirectional_search(grid, start, goal, guide, incumbent, False, guide_first)
    if path is None:
        return None
    stats["hop_expansions"] = hop_exp
    stats["guide_cells"] = guide_cells
    stats["work"] = stats["expanded"] + hop_exp + guide_cells
    return {"cost": cost, "path": path, "expanded": stats["expanded"], "work": stats["work"],
            "touched": stats["touched"], "stats": stats}
