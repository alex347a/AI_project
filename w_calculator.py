import json, time
from llmastar.pather.llm_a_star.fair_core import Grid, grid_astar, build_guide

env = json.load(open("dataset/environment_50_30.json"))[0]
hb, vb = env["horizontal_barriers"], env["vertical_barriers"]
rx = [env["range_x"][0], env["range_x"][1] - 1]      # decremented ranges, as the package uses
ry = [env["range_y"][0], env["range_y"][1] - 1]
moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
grid = Grid(rx, ry, hb, vb, moves)
s, g = grid.index(*env["start_goal"][0][0]), grid.index(*env["start_goal"][0][1])

REPS = 200
t = time.perf_counter()
for _ in range(REPS):
    r = grid_astar(grid, s, g)
per_expansion = (time.perf_counter() - t) / REPS / r["expanded"]

cells = r["path"]                                        # a full start->goal path, like the real incumbent
t = time.perf_counter()
for _ in range(REPS):
    guide = build_guide(grid, cells, 2)
reached = len(guide) - guide.count(3)
per_cell = (time.perf_counter() - t) / REPS / reached

print("w =", per_cell / per_expansion)