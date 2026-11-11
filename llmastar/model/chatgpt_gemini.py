import os
import httpx
from openai import OpenAI

class ChatGPT:
    def __init__(self, method, sysprompt, example):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it first, e.g.  export GEMINI_API_KEY=...  "
                "(the Table 1 statistics can be recomputed without any key, see README)."
            )
        # SSL verification stays ON by default; set LLMASTAR_INSECURE_SSL=1 only behind a
        # proxy that intercepts HTTPS.
        http_client = httpx.Client(verify=os.getenv("LLMASTAR_INSECURE_SSL") != "1")
        self.client = OpenAI(api_key=api_key,
                     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                     http_client=http_client)
        
        # Active Gemini model ID
        self.model_name = "gemini-3.5-flash-lite"

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