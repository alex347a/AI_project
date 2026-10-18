import os
import openai
from llmastar.pather import AStar, LLMAStar

# Explicitly ensure client uses environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL")

# Define sample query matching dataset structure
query = {
    "start": [5, 5], 
    "goal": [27, 15], 
    "size": [51, 31],
    "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
    "vertical_barriers": [[25, 10, 22]],
    "range_x": [0, 51], 
    "range_y": [0, 31]
}

# Run Classical A*
print("Running A* baseline...")
astar_result = AStar().searching(query=query, filepath='astar.png')

# Run LLM-A* (using Groq model)
print("Running LLM-A*...")
llm_result = LLMAStar(llm='gpt', prompt='standard').searching(query=query, filepath='llm.png')

print("Pathfinding complete!")