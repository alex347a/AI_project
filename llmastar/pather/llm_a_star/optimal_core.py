"""
optimal_core.py -- dependency-free core of the optimal unidirectional LLM-A*.

Everything works on a flat integer grid (index = (y - y0) * W + (x - x0)), so it can be tested
without shapely, matplotlib or an LLM.

Guarantee
---------
The returned path is optimal for the 8-connected (or 4-connected) grid graph, whatever the LLM
waypoints are. The LLM plan is used in two ways that cannot hurt correctness:

  1. Incumbent bound: the waypoints are stitched into a feasible path with short A* hops.
     Its cost is an upper bound, so children with f >= cost are never pushed. If the plan is
     already optimal, the search just proves it and returns the plan.
  2. Guide field: a bounded BFS distance-to-plan table that breaks ties between nodes with
     equal f, so the search prefers cells near the LLM route.

Collision semantics match the shapely-based LLMAStar.is_collision: an edge collides iff one of
its endpoints lies on a barrier segment or on the border lines (barriers sit on integer
coordinates and every move is a unit step, so touching can only happen at an endpoint).

  solve_counted     forward A* with the incumbent bound and the guide tie-break. Also counts the
                    work done outside the main search: incumbent hop expansions and guide cells.
"""
import heapq
import math
from array import array
from collections import deque

INF = math.inf
SQRT2 = math.sqrt(2.0)
EPS = 1e-9
_push, _pop = heapq.heappush, heapq.heappop


# --------------------------------------------------------------------------- heuristics
def _octile(x, y, tx, ty):
    dx = abs(x - tx)
    dy = abs(y - ty)
    return dx + dy + (SQRT2 - 2.0) * (dx if dx < dy else dy)


def _manhattan(x, y, tx, ty):
    return abs(x - tx) + abs(y - ty)


def _euclid(x, y, tx, ty):
    return math.hypot(x - tx, y - ty)


_FOUR = {(1, 0), (-1, 0), (0, 1), (0, -1)}
_DIAG = {(1, 1), (1, -1), (-1, 1), (-1, -1)}


def _as_int(v, what):
    f = float(v)
    if not f.is_integer():
        raise ValueError(f"{what} must be an integer coordinate, got {v!r}")
    return int(f)


# --------------------------------------------------------------------------- grid
class Grid:
    """Blocked-cell bytearray plus move table. Ranges are the *decremented* ones used by is_collision."""

    def __init__(self, range_x, range_y, horizontal_barriers, vertical_barriers, motions):
        self.x0, self.x1 = _as_int(range_x[0], "range_x"), _as_int(range_x[1], "range_x")
        self.y0, self.y1 = _as_int(range_y[0], "range_y"), _as_int(range_y[1], "range_y")
        self.W = self.x1 - self.x0 + 1
        self.H = self.y1 - self.y0 + 1
        if self.W < 3 or self.H < 3:
            raise ValueError("map is too small")
        W, H = self.W, self.H
        self.blocked = blocked = bytearray(W * H)
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1

        # border lines are barriers (the original adds them as segments in is_collision)
        for x in range(x0, x1 + 1):
            blocked[x - x0] = 1
            blocked[(H - 1) * W + x - x0] = 1
        for y in range(y0, y1 + 1):
            blocked[(y - y0) * W] = 1
            blocked[(y - y0) * W + W - 1] = 1

        # horizontal barrier: (y, xa, xb) -> segment (xa, y)-(xb, y)
        for h in horizontal_barriers:
            y = _as_int(h[0], "horizontal barrier y")
            if not y0 <= y <= y1:
                continue
            lo, hi = sorted((float(h[1]), float(h[2])))
            for x in range(max(x0, math.ceil(lo)), min(x1, math.floor(hi)) + 1):
                blocked[(y - y0) * W + x - x0] = 1
        # vertical barrier: (x, ya, yb) -> segment (x, ya)-(x, yb)
        for v in vertical_barriers:
            x = _as_int(v[0], "vertical barrier x")
            if not x0 <= x <= x1:
                continue
            lo, hi = sorted((float(v[1]), float(v[2])))
            for y in range(max(y0, math.ceil(lo)), min(y1, math.floor(hi)) + 1):
                blocked[(y - y0) * W + x - x0] = 1

        ms = {(int(a), int(b)) for a, b in motions}
        if any(max(abs(a), abs(b)) != 1 for a, b in ms):
            raise ValueError("only unit moves are supported (the blocked border ring relies on it)")
        self.moves = [(int(a), int(b), int(b) * W + int(a), math.hypot(a, b)) for a, b in motions]
        if ms == _FOUR | _DIAG:
            self.h = _octile
        elif ms == _FOUR:
            self.h = _manhattan
        else:
            self.h = _euclid  # always admissible and consistent when move cost = Euclidean length

    # ---- coordinate helpers
    def index(self, x, y):
        x, y = int(round(x)), int(round(y))
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            raise ValueError(f"({x}, {y}) is outside the map")
        return (y - self.y0) * self.W + (x - self.x0)

    def xy(self, i):
        y, x = divmod(i, self.W)
        return (x + self.x0, y + self.y0)

    def waypoint_indices(self, points):
        """Round, then silently drop waypoints that are outside the map or on a blocked cell."""
        out = []
        for p in points:
            try:
                i = self.index(p[0], p[1])
            except (ValueError, TypeError, IndexError):
                continue
            if not self.blocked[i] and (not out or out[-1] != i):
                out.append(i)
        return out

    def path_cost(self, path):
        return sum(math.dist(self.xy(a), self.xy(b)) for a, b in zip(path, path[1:]))


# --------------------------------------------------------------------------- LLM-derived helpers
def build_guide(grid, cells, radius):
    """Bounded multi-source BFS over free cells: guide[i] = steps to the plan (radius+1 if farther)."""
    N = grid.W * grid.H
    cap = radius + 1
    if cap >= 65535:
        raise ValueError("guide radius too large")
    guide = array("H", [cap]) * N
    dq = deque()
    for c in cells:
        if guide[c] != 0:
            guide[c] = 0
            dq.append(c)
    offs = [m[2] for m in grid.moves]
    blocked = grid.blocked
    while dq:
        i = dq.popleft()
        d = guide[i]
        if d >= radius:
            continue
        for off in offs:
            n = i + off
            if blocked[n] or guide[n] <= d + 1:
                continue
            guide[n] = d + 1
            dq.append(n)
    return guide


def _rebuild(par, i):
    path = [i]
    while par[path[-1]] != path[-1]:
        path.append(par[path[-1]])
    return path[::-1]


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