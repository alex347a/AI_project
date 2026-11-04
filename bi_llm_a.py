"""
Bidirectional LLM-A* (fast version, no IDA* phase).

Drop this file next to llm_a_star.py (llmastar/pather/llm_a_star/) and use:

    from llmastar.pather.llm_a_star.bi_llm_a_star_fast import BiLLMAStarFast
    BiLLMAStarFast(llm='gpt', prompt='standard').searching(query=query, filepath='bi.png')

One search grows from the start and one from the goal. Both use the LLM waypoint list
(the goal side uses it reversed) and the same "distance to goal + distance to current
waypoint" heuristic as LLMAStar. This heuristic is not admissible, so the path is valid
but not guaranteed to be the shortest.
"""
import copy
import heapq
import math

from .llm_a_star import LLMAStar


class _Side:
    """Search state for one direction (forward or backward)."""

    def __init__(self, origin, goal, targets):
        self.origin = origin
        self.goal = goal
        self.targets = targets          # waypoint list in this side's travel order
        self.i = 1
        self.target = targets[1]
        self.open = []
        self.closed = set()
        self.order = []                 # expansion order, used for plotting
        self.g = {origin: 0.0}
        self.parent = {origin: origin}


class BiLLMAStarFast(LLMAStar):
    """LLM-A* that searches from both the start and the goal."""

    # ---------- helpers ----------
    def cost(self, s_start, s_goal):
        """Memoised edge cost (collision checks are the slowest part of the search)."""
        key = (s_start, s_goal)
        c = self._cost_cache.get(key)
        if c is None:
            c = super().cost(s_start, s_goal)
            self._cost_cache[key] = c
            self._cost_cache[(s_goal, s_start)] = c
        return c

    @staticmethod
    def _h_side(side, s):
        return (math.hypot(side.goal[0] - s[0], side.goal[1] - s[1])
                + math.hypot(side.target[0] - s[0], side.target[1] - s[1]))

    def _push(self, side, s):
        heapq.heappush(side.open, (side.g[s] + self._h_side(side, s), s))

    def _advance_target(self, side):
        side.i += 1
        if side.i < len(side.targets):
            side.target = side.targets[side.i]
        side.open = [(side.g[s] + self._h_side(side, s), s) for _, s in side.open]
        heapq.heapify(side.open)

    def _path_length(self, path):
        return sum(self._euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    # ---------- phase 1: bidirectional LLM-A* ----------
    def _bidirectional(self):
        targets = list(self.target_list)
        fwd = _Side(self.s_start, self.s_goal, targets)
        bwd = _Side(self.s_goal, self.s_start, targets[::-1])
        self._push(fwd, self.s_start)
        self._push(bwd, self.s_goal)

        mu, meet = math.inf, None

        while fwd.open and bwd.open:
            # expand the side with the smaller frontier
            side, other = (fwd, bwd) if len(fwd.open) <= len(bwd.open) else (bwd, fwd)
            f, s = heapq.heappop(side.open)
            if s in side.closed:
                continue
            if f >= mu:                 # nothing left in this direction can beat the meeting point
                break
            side.closed.add(s)
            side.order.append(s)

            for s_n in self.get_neighbor(s):
                if s_n == side.target and side.origin != side.target and side.goal != side.target:
                    self._advance_target(side)
                if s_n in side.closed:
                    continue
                new_cost = side.g[s] + self.cost(s, s_n)
                if new_cost < side.g.get(s_n, math.inf):
                    side.g[s_n] = new_cost
                    side.parent[s_n] = s
                    self._push(side, s_n)
                    if s_n in other.g and new_cost + other.g[s_n] < mu:
                        mu, meet = new_cost + other.g[s_n], s_n

        if meet is None:
            raise RuntimeError("Bidirectional search failed: the two searches never met (no path found).")

        left = [meet]
        while left[-1] != self.s_start:
            left.append(fwd.parent[left[-1]])
        right = [meet]
        while right[-1] != self.s_goal:
            right.append(bwd.parent[right[-1]])
        path = left[::-1] + right[1:]
        return path, fwd, bwd

    # ---------- public entry point ----------
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

        path, fwd, bwd = self._bidirectional()
        result = {
            "operation": len(fwd.order) + len(bwd.order),
            "storage": len(fwd.g) + len(bwd.g),
            "length": self._path_length(path),
            "llm_output": self.target_list,
        }
        print(result)
        self.plot.animation(path, fwd.order + bwd.order, show, "Bi LLM-A*", self.filepath)
        return result
