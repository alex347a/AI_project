import os
import copy
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from llmastar.pather import AStar, LLMAStar
from llmastar.pather.llm_a_star.bi_a_star import BiAStar
from llmastar.pather.llm_a_star.bi_llm_a import BiLLMAStarFast

query = {"start": [5, 5], "goal": [27, 15], "size": [51, 31],
         "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
         "vertical_barriers": [[25, 10, 22]],
         "range_x": [0, 51], "range_y": [0, 31]}

# Each search gets its own copy, because the original classes edit range_x/range_y in place
astar    = AStar().searching(query=copy.deepcopy(query), filepath='astar.png')
bi_astar = BiAStar().searching(query=copy.deepcopy(query), filepath='bi_astar.png')
llm      = LLMAStar(llm='gpt', prompt='standard').searching(query=copy.deepcopy(query), filepath='llm.png')
bi_llm   = BiLLMAStarFast(llm='gpt', prompt='standard').searching(query=copy.deepcopy(query), filepath='bi_llm.png')

print("A*        :", astar)
print("Bi A*     :", bi_astar)
print("LLM-A*    :", llm)
print("Bi LLM-A* :", bi_llm)
