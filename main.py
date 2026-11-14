import os
import copy
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not found: check your .env file"

from llmastar.pather import AStar, LLMAStar
from llmastar.pather.llm_a_star.bi_a_star import BiAStar
from llmastar.pather.llm_a_star.bi_llm_a import BiLLMAStarFast

query = {"start": [5, 5], "goal": [27, 15], "size": [51, 31],
         "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
         "vertical_barriers": [[25, 10, 22]],
         "range_x": [0, 51], "range_y": [0, 31]}

# Each search gets its own copy, because the original classes edit range_x/range_y in place
astar    = AStar().searching(query=copy.deepcopy(query), filepath='testResultImages/astar.png')
bi_astar = BiAStar().searching(query=copy.deepcopy(query), filepath='testResultImages/bi_astar.png')
llm      = LLMAStar(llm='gpt', prompt='standard').searching(query=copy.deepcopy(query), filepath='testResultImages/llm.png')
bi_llm   = BiLLMAStarFast(llm='gpt', prompt='standard').searching(query=copy.deepcopy(query), filepath='testResultImages/bi_llm.png')

print("A*        :", astar)
print("Bi A*     :", bi_astar)
print("LLM-A*    :", llm)
print("Bi LLM-A* :", bi_llm)

# ---- ablation: what does each use of the LLM plan contribute?
# The LLM output varies between calls, so each config gets several runs.
N_RUNS = 3
configs = [("no LLM info", dict(use_incumbent=False, use_guide=False)),
           ("incumbent only", dict(use_incumbent=True, use_guide=False)),
           ("guide only", dict(use_incumbent=False, use_guide=True)),
           ("both", dict())]

print("\nAblation (operation = total expansions, storage = total touched, length)")
for name, kw in configs:
    for run in range(N_RUNS):
        r = BiLLMAStarFast(llm='gpt', prompt='standard', **kw).searching(
            query=copy.deepcopy(query),
            filepath=f'testResultImages/abl_{name.replace(" ", "_")}_{run}.png',
            show=False, plot=False)
        s = r['stats']
        print(f"{name:15s} run {run}: op={r['operation']:4d} storage={r['storage']:4d} "
              f"len={r['length']:.3f} | search={s['expanded']} incumbent={s['incumbent_expanded']} "
              f"pruned={s['pruned']} inc_cost={s['incumbent_cost']}")