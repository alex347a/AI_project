import copy
from llmastar.pather import AStar
from llmastar.pather.llm_a_star.bi_llm_a import BiLLMAStarFast

class NoLLM(BiLLMAStarFast):
    def __init__(self, waypoints, **kw):
        # skip the parent __init__, which creates the ChatGPT clients
        self.waypoints = waypoints
        self.use_incumbent = kw.get("use_incumbent", True)
        self.use_guide = kw.get("use_guide", True)
        self.guide_radius = 8
        self.hop_cap_factor = 20
        self.guide_first = True

    def _initialize_llm_paths(self):
        self.target_list = [self.s_start, *self.waypoints, self.s_goal]

query = {"start": [5, 5], "goal": [27, 15], "size": [51, 31],
         "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
         "vertical_barriers": [[25, 10, 22]],
         "range_x": [0, 51], "range_y": [0, 31]}

ref = AStar().searching(query=copy.deepcopy(query), filepath="astar.png")

plans = {"none": [], "exact": [(25, 9), (26, 10)], "garbage": [(40, 5), (40, 28), (8, 28)]}
for name, wps in plans.items():
    r = NoLLM(wps).searching(query=copy.deepcopy(query), filepath=f"{name}.png", show=False)
    print(name, r["length"], r["operation"], r["stats"]["incumbent_optimal"])
    assert abs(r["length"] - ref["length"]) < 1e-6   # must match plain A*