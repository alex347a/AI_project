import os
import time
from google import genai
from google.genai import types


class ChatGPT:
    def __init__(self, method, sysprompt, example):
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=60000),  # 60 s, en ms
        )
        # Modele modifiable sans toucher au code : $env:GEMINI_MODEL="gemini-3.5-flash"
        self.model_id = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.debug = os.environ.get("DEBUG_LLM") == "1"

        self.id = 0
        self.chat_history = [{"role": "system", "content": sysprompt}]

        if example:
            self.prompt = sysprompt + '\nFollow these examples delimited with “”" as a guide.\n'
            keys = list(example.keys())
            for key in keys:
                input_str = example[key]
                index = input_str.find("\n") + 1
                self.prompt += f'“”"\nUser: {input_str[:index - 1]}\nAssistant: {input_str[index:]}“”"\n'
        else:
            self.prompt = sysprompt

    def _generate(self, full_prompt, max_tokens):
        # Pas de stop_sequences : sinon la reponse est coupee au premier retour a la ligne
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.0,
        )
        for attempt in range(8):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=config,
                )
                text = response.text or ""
                if self.debug:
                    print("REPONSE LLM:", repr(text))
                return text
            except Exception as e:
                print(f"Erreur API (tentative {attempt + 1}/5) : {str(e)[:300]}")
                msg = str(e)
                # Quota atteint ou mauvais modele : inutile de reessayer ici
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg or "NOT_FOUND" in msg:
                    raise
                if attempt < 7:
                    time.sleep(20 * (attempt + 1))
                else:
                    return ""

    def ask(self, prompt, stop=None, max_tokens=1000):
        return self._generate(self.prompt + "\n\n" + prompt, max_tokens)

    def chat(self, query, prompt="", stop=None, max_tokens=1000):
        return self._generate(
            self.prompt + f"\nUser: {query}\nAssistant: {prompt}", max_tokens
        )

    def chat_with_image(self, chat_history, stop=None, max_tokens=100):
        pass