import os
import httpx
from groq import Groq

# Place your Groq API key here
GROQ_API_KEY = "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"

class ChatGPT:
    def __init__(self, method, sysprompt, example):
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or GROQ_API_KEY
        
        # Bypass local network SSL interception if present
        http_client = httpx.Client(verify=False)
        self.client = Groq(api_key=api_key, http_client=http_client)
        
        # Active Groq model ID
        self.model_name = "openai/gpt-oss-20b"

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
        return response.choices[0].message.content.strip()
    
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
        return response.choices[0].message.content.strip()
    
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