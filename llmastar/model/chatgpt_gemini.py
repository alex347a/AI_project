import os

from openai import OpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _resolve_config():
    """Return (api_key, base_url, model) from environment variables.

    Any OpenAI-compatible endpoint can be used:
      LLM_API_KEY   (or OPENAI_API_KEY)  API key
      LLM_BASE_URL  (or OPENAI_BASE_URL) endpoint; unset -> the default OpenAI endpoint
      LLM_MODEL                          model name; default gpt-4o-mini
    Fallback used for our Gemini experiments: with only GEMINI_API_KEY set, the Gemini
    OpenAI-compatible endpoint and gemini-3.5-flash-lite are used by default.
    """
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return key, os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL", "gpt-4o-mini")
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return (key, os.getenv("LLM_BASE_URL", GEMINI_BASE_URL),
                os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"))
    raise RuntimeError(
        "No API key found. Set LLM_API_KEY (or OPENAI_API_KEY), optionally LLM_BASE_URL and "
        "LLM_MODEL (default model: gpt-4o-mini). For Gemini, set GEMINI_API_KEY instead. "
        "(The Table 1 statistics can be recomputed without any key, see README.)"
    )


class ChatGPT:
    def __init__(self, method, sysprompt, example):
        api_key, base_url, self.model_name = _resolve_config()
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        # SSL verification stays ON by default; set LLMASTAR_INSECURE_SSL=1 only behind a
        # proxy that intercepts HTTPS.
        if os.getenv("LLMASTAR_INSECURE_SSL") == "1":
            import httpx
            client_kwargs["http_client"] = httpx.Client(verify=False)
        self.client = OpenAI(**client_kwargs)

        self.id = 0
        self.chat_history = [
            {"role": "system", "content": sysprompt}
        ]
        if example:
            self.prompt = sysprompt + f'\nFollow these examples delimited with “”" as a guide.\n'
            keys = list(example.keys())
            for key in keys:
                input_text = example[key]
                index = input_text.find("\n") + 1
                self.prompt += f'“”"\nUser: {input_text[:index - 1]}\nAssistant: {input_text[index:]}“”"\n'
                self.chat_history.append({"role": "user", "content": input_text[:index - 1]})
                self.chat_history.append({"role": "assistant", "content": input_text[index:]})
        else:
            self.prompt = sysprompt

    def ask(self, prompt, stop=None, max_tokens=1000, temperature=0):
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop and stop != ["\n"]:
            kwargs["stop"] = stop

        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
    
    def chat(self, query, prompt="", stop=None, max_tokens=1000, temperature=0):
        content = f"{prompt}\n{query}".strip() if prompt else query
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop and stop != ["\n"]:
            kwargs["stop"] = stop

        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()
    
    def chat_with_image(self, chat_history, stop=None, max_tokens=1000, temperature=0):
        kwargs = {
            "model": self.model_name,
            "messages": chat_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop and stop != ["\n"]:
            kwargs["stop"] = stop

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content