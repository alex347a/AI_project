"""
Optimal bidirectional LLM-A*.

Drop this file and optimal_core.py into llmastar/pather/llm_a_star/ and use:

    from llmastar.pather.llm_a_star.bi_llm_a import BiLLMAStarFast
    BiLLMAStarFast(llm='gpt', prompt='standard').searching(query=query, filepath='bi_llm.png')

The path is optimal on the grid graph regardless of what the LLM returns. The LLM plan only
(1) gives an upper bound that prunes the heap and (2) breaks ties between equal-f nodes.
Toggle them with use_incumbent / use_guide / guide_first for ablations.

"operation" and "storage" include the work spent building the incumbent from the LLM waypoints,
not just the final bidirectional search. The split is in result["stats"].
"""
import copy
import time

from .llm_a_star import LLMAStar
from .optimal_core import Grid, solve


class BiLLMAStarFast(LLMAStar):
    def __init__(self, llm='gpt', prompt='standard', use_incumbent=True, use_guide=True,
                 guide_radius=8, hop_cap_factor=20, guide_first=True):
        super().__init__(llm=llm, prompt=prompt)
        self.use_incumbent = use_incumbent
        self.use_guide = use_guide
        self.guide_radius = guide_radius
        self.hop_cap_factor = hop_cap_factor
        self.guide_first = guide_first   # guide ranks ahead of h among equal-f nodes

    def searching(self, query, filepath='temp.png', show=True, plot=True):
        """
        :param show: open the matplotlib window
        :param plot: set False to skip plotting (and the expansion-order bookkeeping) entirely
        """
        self.filepath = filepath
        query = copy.deepcopy(query)   # _initialize_parameters edits range_x/range_y in place
        print(query)

        t = time.perf_counter()
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        llm_seconds = time.perf_counter() - t

        # self.range_x / range_y are the decremented ranges the original is_collision uses
        grid = Grid(self.range_x, self.range_y, self.horizontal_barriers,
                    self.vertical_barriers, self.u_set)
        start = grid.index(*self.s_start)
        goal = grid.index(*self.s_goal)
        waypoints = grid.waypoint_indices(self.target_list[1:-1])

        res = solve(grid, start, goal, waypoints,
                    use_incumbent=self.use_incumbent, use_guide=self.use_guide,
                    guide_radius=self.guide_radius, hop_cap_factor=self.hop_cap_factor,
                    record_order=plot, guide_first=self.guide_first)
        if res["path"] is None:
            raise RuntimeError("No path exists between start and goal.")

        stats = res["stats"]
        stats["llm_seconds"] = llm_seconds
        path = [grid.xy(i) for i in res["path"]]
        result = {
            "operation": stats["total_expanded"],   # incumbent hops + bidirectional search
            "storage": stats["total_touched"],      # upper bound: hop and search nodes may overlap
            "length": res["cost"],
            "llm_output": self.target_list,
            "stats": stats,
        }
        print(result)

        if plot:
            fwd, bwd = ([grid.xy(i) for i in o] for o in res["orders"])
            visited = []
            for k in range(max(len(fwd), len(bwd))):   # interleave so the animation shows both searches
                if k < len(fwd):
                    visited.append(fwd[k])
                if k < len(bwd):
                    visited.append(bwd[k])
            self.plot.animation(path, visited, show, "Bi LLM-A*", self.filepath)
        return result