"""
Optimal unidirectional LLM-A*.

Drop this file and optimal_core.py into llmastar/pather/llm_a_star/ and use:

    from llmastar.pather.llm_a_star.uni_llm_a import UniLLMAStarFast
    UniLLMAStarFast(llm='gpt', prompt='standard').searching(query=query, filepath='uni_llm.png')

The path is optimal on the grid graph regardless of what the LLM returns. The LLM plan only
(1) gives an upper bound that prunes the heap and (2) breaks ties between equal-f nodes.
"""
import copy
import time

from .llm_a_star import LLMAStar
from .optimal_core import Grid, solve_counted


class UniLLMAStarFast(LLMAStar):
    def __init__(self, llm='gpt', prompt='standard', use_incumbent=True, use_guide=True,
                 guide_radius=2, hop_cap_factor=20, guide_first=True):
        super().__init__(llm=llm, prompt=prompt)
        self.use_incumbent = use_incumbent
        self.use_guide = use_guide
        self.guide_radius = guide_radius
        self.hop_cap_factor = hop_cap_factor
        self.guide_first = guide_first   # guide ranks ahead of h among equal-f nodes

    def searching(self, query, filepath='temp.png', show=True, plot=True):
        """
        :param show: open the matplotlib window
        :param plot: set False to skip plotting
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

        t = time.perf_counter()
        res = solve_counted(grid, start, goal, waypoints,
                            use_incumbent=self.use_incumbent, use_guide=self.use_guide,
                            guide_radius=self.guide_radius, hop_cap_factor=self.hop_cap_factor,
                            guide_first=self.guide_first)
        search_seconds = time.perf_counter() - t
        if res is None:
            raise RuntimeError("No path exists between start and goal.")

        stats = res["stats"]
        stats["llm_seconds"] = llm_seconds
        stats["search_seconds"] = search_seconds
        path = [grid.xy(i) for i in res["path"]]
        result = {
            "operation": res["expanded"],    # nodes expanded in the main search
            "storage": res["storage"],       # touched nodes + incumbent path + guide cells
            "length": res["cost"],
            "llm_output": self.target_list,
            "stats": stats,
        }
        print(result)

        if plot:
            # solve_counted does not record expansion order, so only the final path is drawn
            self.plot.animation(path, path, show, "Uni LLM-A*", self.filepath)
        return result