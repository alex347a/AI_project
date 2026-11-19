"""
Bidirectional A* (no LLM, optimal).

Drop this file next to llm_a_star.py (llmastar/pather/llm_a_star/) and use:

    from llmastar.pather.llm_a_star.bi_a_star import BiAStar
    BiAStar().searching(query=query, filepath='bi_astar.png')

One search grows from the start and one from the goal, both with straight-line
distance to the other end as an admissible heuristic. They stop when neither side
can still find a shorter path, so the result is always optimal.
"""
import copy
import heapq
import math

from .llm_a_star import LLMAStar

EPS = 1e-9


class _Side:
    """Search state for one direction (forward or backward)."""

    def __init__(self, origin, goal):
        self.origin = origin
        self.goal = goal                # the point this side's heuristic aims at
        self.open = []
        self.closed = set()
        self.order = []                 # expansion order, used for plotting
        self.g = {origin: 0.0}
        self.parent = {origin: origin}


class BiAStar(LLMAStar):
    """Plain A* (no LLM) that searches from both the start and the goal and stays optimal."""

    def __init__(self):
        pass                            # no LLM, so no API key is needed

    def _initialize_llm_paths(self):
        # no LLM: the only "waypoints" are the start and the goal
        self.target_list = [self.s_start, self.s_goal]

    def cost(self, s_start, s_goal):
        """Memoised edge cost (collision checks are the slowest part of the search)."""
        key = (s_start, s_goal)
        c = self._cost_cache.get(key)
        if c is None:
            c = super().cost(s_start, s_goal)
            self._cost_cache[key] = c
            self._cost_cache[(s_goal, s_start)] = c
        return c

    # ---------- heuristic / queue ----------
    @staticmethod
    def _h(side, s):
        return math.hypot(side.goal[0] - s[0], side.goal[1] - s[1])

    def _wp_dist(self, s):
        return min(math.hypot(w[0] - s[0], w[1] - s[1]) for w in self.target_list)

    def _push(self, side, s):
        f = side.g[s] + self._h(side, s)
        # (rounded f, waypoint distance) -> LLM only breaks ties, never changes optimality
        heapq.heappush(side.open, (round(f, 9), self._wp_dist(s), s))

    def _path_length(self, path):
        return sum(self._euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    # ---------- LLM upper bound ----------
    def _grid_segment(self, a, b):
        """
        Grid-move path (the same 8 moves the search uses) from a to b: diagonal steps
        then straight steps, or the reverse. Returns (nodes, cost), or (None, inf)
        if both orders hit an obstacle.
        """
        if any(float(v) != int(v) for v in (*a, *b)):
            return None, math.inf
        a = (int(a[0]), int(a[1]))
        b = (int(b[0]), int(b[1]))
        dx, dy = b[0] - a[0], b[1] - a[1]
        sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        d = min(abs(dx), abs(dy))
        diag, straight = (sx, sy), ((sx, 0) if abs(dx) > abs(dy) else (0, sy))
        n_straight = max(abs(dx), abs(dy)) - d
        best = (None, math.inf)
        for steps in ([diag] * d + [straight] * n_straight, [straight] * n_straight + [diag] * d):
            nodes, total, cur = [a], 0.0, a
            for m in steps:
                nxt = (cur[0] + m[0], cur[1] + m[1])
                c = self.cost(cur, nxt)
                if c == math.inf:
                    total = math.inf
                    break
                total += c
                nodes.append(nxt)
                cur = nxt
            if total < best[1]:
                best = (nodes, total)
        return best

    def _waypoint_path(self):
        """Valid grid path through the LLM waypoints, or (None, inf) if any leg is blocked."""
        path, total = [self.target_list[0]], 0.0
        for a, b in zip(self.target_list, self.target_list[1:]):
            nodes, c = self._grid_segment(a, b)
            if nodes is None:
                return None, math.inf
            path += nodes[1:]
            total += c
        return path, total

    # ---------- bidirectional search ----------
    def _bidirectional(self):
        fwd = _Side(self.s_start, self.s_goal)
        bwd = _Side(self.s_goal, self.s_start)
        self._push(fwd, self.s_start)
        self._push(bwd, self.s_goal)

        best_path, mu = self._waypoint_path()
        meet = None

        while fwd.open and bwd.open:
            # stop when neither side can still produce a path shorter than mu
            if max(fwd.open[0][0], bwd.open[0][0]) >= mu - EPS:
                break

            side, other = (fwd, bwd) if len(fwd.open) <= len(bwd.open) else (bwd, fwd)
            f, _, s = heapq.heappop(side.open)
            if s in side.closed:
                continue
            side.closed.add(s)
            side.order.append(s)

            for s_n in self.get_neighbor(s):
                if s_n in side.closed:
                    continue
                c = self.cost(s, s_n)
                if c == math.inf:
                    continue
                new_cost = side.g[s] + c
                if new_cost + self._h(side, s_n) >= mu - EPS:
                    continue            # cannot beat the best path already known
                if new_cost < side.g.get(s_n, math.inf):
                    side.g[s_n] = new_cost
                    side.parent[s_n] = s
                    self._push(side, s_n)
                    if s_n in other.g and new_cost + other.g[s_n] < mu - EPS:
                        mu, meet = new_cost + other.g[s_n], s_n

        if meet is None:
            if best_path is None:
                raise RuntimeError("No path found between start and goal.")
            return best_path, fwd, bwd, True   # the LLM polyline was already optimal

        left = [meet]
        while left[-1] != self.s_start:
            left.append(fwd.parent[left[-1]])
        right = [meet]
        while right[-1] != self.s_goal:
            right.append(bwd.parent[right[-1]])
        return left[::-1] + right[1:], fwd, bwd, False

    def searching(self, query, filepath='temp.png', show=True):
        """
        :param show: open the matplotlib window (False just saves the image)
        """
        self.filepath = filepath
        query = copy.deepcopy(query)    # _initialize_parameters edits range_x/range_y in place
        print(query)
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        self._cost_cache = {}

        path, fwd, bwd, from_llm = self._bidirectional()
        result = {
            "operation": len(fwd.order) + len(bwd.order),
            "storage": len(fwd.g) + len(bwd.g),
            "length": self._path_length(path),
        }
        print(result)
        self.plot.animation(path, fwd.order + bwd.order, show, "Bi A*", self.filepath)
        return result
