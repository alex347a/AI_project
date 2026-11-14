"""
optimal_core.py -- dependency-free core of the optimal bidirectional LLM-A*.

Everything works on a flat integer grid (index = (y - y0) * W + (x - x0)), so it can be tested
without shapely, matplotlib or an LLM.

Guarantees
----------
The returned path is optimal for the 8-connected (or 4-connected) grid graph, whatever the LLM
waypoints are. The LLM plan is used in two ways that cannot hurt correctness:

  1. Incumbent bound: the waypoints are stitched into a feasible path with short A* hops.
     Its cost seeds the upper bound `mu`, so children with f >= mu are never pushed. If the
     plan is already optimal, the search just proves it and returns the plan.
  2. Guide field: a bounded BFS distance-to-plan table that breaks ties between nodes with
     equal f, so the search prefers cells near the LLM route.

Collision semantics match the shapely-based LLMAStar.is_collision: an edge collides iff one of
its endpoints lies on a barrier segment or on the border lines (barriers sit on integer
coordinates and every move is a unit step, so touching can only happen at an endpoint).

Work accounting: `expanded` / `touched` cover only the bidirectional search. The A* hops that
build the incumbent are counted separately (`incumbent_expanded` / `incumbent_touched`), and
`total_expanded` / `total_touched` are the sums.
"""
import heapq
import math
from array import array
from collections import deque
from time import perf_counter

INF = math.inf
SQRT2 = math.sqrt(2.0)
EPS = 1e-9
_heappush, _heappop = heapq.heappush, heapq.heappop


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

    def is_blocked_xy(self, x, y):
        """Cells outside the range count as free (only the border ring blocks)."""
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return False
        return bool(self.blocked[(y - self.y0) * self.W + (x - self.x0)])

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
def astar_hop(grid, a, b, max_expansions):
    """Small unidirectional A* between two waypoints.

    Returns (path | None, expanded, touched) so the caller can account for the work done.
    """
    W, blocked, moves, h = grid.W, grid.blocked, grid.moves, grid.h
    by, bx = divmod(b, W)
    ay, ax = divmod(a, W)
    g = {a: 0.0}
    par = {a: a}
    closed = set()
    heap = [(h(ax, ay, bx, by), a)]
    expanded = 0
    while heap:
        _, i = _heappop(heap)
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
                _heappush(heap, (ng + h(x + dx, y + dy, bx, by), n))
    return None, expanded, len(g)


def build_incumbent(grid, start, goal, waypoints, cap_factor=20, work=None):
    """Stitch the waypoints into a feasible start->goal path. Returns (path, cost) or None.

    `work`, if given, is a dict with "expanded" and "touched" counters that accumulate the
    cost of every hop search (including failed ones).
    """
    path = [start]
    cur = start
    W = grid.W
    for w in list(waypoints) + [goal]:
        if w == cur or grid.blocked[w]:
            continue
        (cy, cx), (wy, wx) = divmod(cur, W), divmod(w, W)
        cap = int(cap_factor * (grid.h(cx, cy, wx, wy) + 10))
        hop, e, t = astar_hop(grid, cur, w, cap)
        if work is not None:
            work["expanded"] += e
            work["touched"] += t
        if hop is None:
            if w == goal:
                return None
            continue  # skip an unreachable or too-expensive waypoint
        path.extend(hop[1:])
        cur = w
    if cur != goal:
        return None
    return path, grid.path_cost(path)


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


# --------------------------------------------------------------------------- the search
def bidirectional_search(grid, start, goal, guide=None, incumbent=None, record_order=False, guide_first=False):
    """
    Optimal bidirectional A*, consistent heuristics, stop when max(minF_fwd, minF_bwd) >= mu.
    incumbent: optional (path, cost) seeding mu. Returns (path | None, cost, stats, (order_f, order_b)).
    """
    t0 = perf_counter()
    W = grid.W
    N = W * grid.H
    blocked, moves, h = grid.blocked, grid.moves, grid.h
    if blocked[start] or blocked[goal]:
        raise ValueError("start or goal lies on a barrier or border")
    stats = {"expanded": 0, "expanded_fwd": 0, "expanded_bwd": 0, "pushes": 0, "touched": 2,
             "peak_open": 2, "pruned": 0}
    orders = ([], [])
    if start == goal:
        stats["seconds"] = perf_counter() - t0
        return [start], 0.0, stats, orders

    sy, sx = divmod(start, W)
    gy, gx = divmod(goal, W)
    mu = INF
    meet = None
    inc_path = None
    if incumbent is not None:
        inc_path, mu = incumbent

    g = (array("d", [INF]) * N, array("d", [INF]) * N)
    par = (array("i", [-1]) * N, array("i", [-1]) * N)
    closed = (bytearray(N), bytearray(N))
    heaps = ([], [])
    tx = (gx, sx)      # each side's target coordinates (fwd -> goal, bwd -> start)
    ty = (gy, sy)
    g[0][start] = 0.0
    g[1][goal] = 0.0
    par[0][start] = start
    par[1][goal] = goal
    h0 = h(sx, sy, gx, gy)
    for hp, node in ((heaps[0], start), (heaps[1], goal)):
        gk0 = guide[node] if guide is not None else 0
        hp.append((round(h0, 9), gk0, h0, node) if guide_first else (round(h0, 9), h0, gk0, node))

    pushes = 0
    pruned = 0
    touched = 2
    peak = 2
    expanded = [0, 0]
    hf, hb = heaps
    cf, cb = closed

    while True:
        while hf and cf[hf[0][3]]:
            _heappop(hf)
        while hb and cb[hb[0][3]]:
            _heappop(hb)
        if not hf or not hb:
            break
        top = hf[0][0] if hf[0][0] > hb[0][0] else hb[0][0]
        if top >= mu - EPS:
            break
        size = len(hf) + len(hb)
        if size > peak:
            peak = size

        d = 0 if len(hf) <= len(hb) else 1
        heap, clo, gd, pd, og = heaps[d], closed[d], g[d], par[d], g[1 - d]
        ttx, tty = tx[d], ty[d]
        i = _heappop(heap)[3]
        clo[i] = 1
        expanded[d] += 1
        if record_order:
            orders[d].append(i)

        gi = gd[i]
        y, x = divmod(i, W)
        for dx, dy, off, c in moves:
            n = i + off
            if blocked[n] or clo[n]:
                continue
            ng = gi + c
            old = gd[n]
            if ng >= old:
                continue
            if old == INF:
                touched += 1
            gd[n] = ng
            pd[n] = i
            on = og[n]
            if on != INF:
                tot = ng + on
                if tot < mu - EPS:
                    mu = tot
                    meet = n
            nh = h(x + dx, y + dy, ttx, tty)
            nf = round(ng + nh, 9)
            if nf >= mu - EPS:
                pruned += 1
                continue            # cannot lead to a path better than mu (h is admissible)
            gk = guide[n] if guide is not None else 0
            _heappush(heap, (nf, gk, nh, n) if guide_first else (nf, nh, gk, n))
            pushes += 1

    stats.update(expanded=expanded[0] + expanded[1], expanded_fwd=expanded[0], expanded_bwd=expanded[1],
                 pushes=pushes, pruned=pruned, touched=touched, peak_open=peak)

    if meet is not None:
        left = [meet]
        while left[-1] != start:
            left.append(par[0][left[-1]])
        right = [meet]
        while right[-1] != goal:
            right.append(par[1][right[-1]])
        path = left[::-1] + right[1:]
    elif inc_path is not None:
        path = list(inc_path)
    else:
        stats["seconds"] = perf_counter() - t0
        return None, INF, stats, orders

    stats["seconds"] = perf_counter() - t0
    return path, grid.path_cost(path), stats, orders


def solve(grid, start, goal, waypoints=(), use_incumbent=True, use_guide=True,
          guide_radius=8, hop_cap_factor=20, record_order=False, guide_first=False):
    """Plan -> incumbent -> guide -> optimal search. Returns a dict with path, cost, stats, orders."""
    t_all = perf_counter()
    incumbent = None
    inc_work = {"expanded": 0, "touched": 0}
    t_inc = 0.0
    if use_incumbent and waypoints:
        t = perf_counter()
        incumbent = build_incumbent(grid, start, goal, waypoints, hop_cap_factor, inc_work)
        t_inc = perf_counter() - t

    guide = None
    t_guide = 0.0
    if use_guide and (incumbent is not None or waypoints):
        t = perf_counter()
        cells = incumbent[0] if incumbent is not None else [start, *waypoints, goal]
        guide = build_guide(grid, cells, guide_radius)
        t_guide = perf_counter() - t

    path, cost, stats, orders = bidirectional_search(grid, start, goal, guide, incumbent, record_order, guide_first)
    stats["incumbent_cost"] = incumbent[1] if incumbent is not None else None
    stats["incumbent_optimal"] = (incumbent is not None and path is not None
                                  and abs(incumbent[1] - cost) <= 1e-7)
    stats["t_incumbent"] = t_inc
    stats["t_guide"] = t_guide
    stats["incumbent_expanded"] = inc_work["expanded"]
    stats["incumbent_touched"] = inc_work["touched"]
    stats["total_expanded"] = stats["expanded"] + inc_work["expanded"]
    stats["total_touched"] = stats["touched"] + inc_work["touched"]
    stats["t_search"] = stats.pop("seconds")
    stats["t_total"] = perf_counter() - t_all
    return {"path": path, "cost": cost, "stats": stats, "orders": orders}