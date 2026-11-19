"""Smoke test: one real call through the model wrapper (needs an API key in the environment).

    python check_llm.py
"""
from llmastar.model import ChatGPT

bot = ChatGPT(method=None, sysprompt="", example=None)
print("model :", bot.model_name)
print("base  :", bot.client.base_url)
print("answer:", bot.ask("Reply with the single word OK.", max_tokens=20))
