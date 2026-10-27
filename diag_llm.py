"""
One-query diagnostic: shows exactly what the LLM call receives and returns inside LLM-A*.
It makes ONE model request (about 1.6K tokens).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

from llmastar.env.search import plotting
plotting.Plotting.animation = lambda self, *a, **k: None   # no drawing

from llmastar import model
from llmastar.pather import LLMAStar

LLM = os.environ.get("DIAG_LLM", "gpt")
PROMPT = os.environ.get("DIAG_PROMPT", "standard")
cls = model.ChatGPT if LLM == "gpt" else model.Llama3
orig_ask = cls.ask
calls = {"n": 0}


def spy(self, *a, **k):
    calls["n"] += 1
    override = os.environ.get("DIAG_MAX_TOKENS")
    if override:
        k["max_tokens"] = int(override)
    shown = {kk: (vv if kk != "prompt" else f"<{len(str(vv))} chars>") for kk, vv in k.items()}
    print(f"\n[ask #{calls['n']}] positional args: {len(a)}, kwargs: {shown}")
    try:
        r = orig_ask(self, *a, **k)
    except Exception as e:
        print(f"[ask #{calls['n']}] RAISED {type(e).__name__}: {e}")
        raise
    text = r if isinstance(r, str) else repr(r)
    print(f"[ask #{calls['n']}] RETURNED {len(text)} chars. First 1500 chars:")
    print(text[:1500] if text else "<EMPTY STRING>")
    return r


cls.ask = spy

query = {
    "start": [5, 5], "goal": [20, 20], "size": [51, 31],
    "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
    "vertical_barriers": [[25, 10, 22]],
    "range_x": [0, 51], "range_y": [0, 31],
}

print(f"Running LLM-A* ({LLM}, {PROMPT}) on one query...")
res = LLMAStar(llm=LLM, prompt=PROMPT).searching(query=query)
print("\nRESULT:", res)
print(f"\nModel was asked {calls['n']} time(s).")