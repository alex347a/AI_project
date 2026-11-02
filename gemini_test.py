import os
from openai import OpenAI, APIStatusError

client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = "gemini-3.5-flash-lite"   # change to a name from the list below if needed

print("Key variable is set:", bool(os.getenv("GEMINI_API_KEY")))

try:
    names = [m.id for m in client.models.list()]
    print(f"\n{len(names)} models available to your key:")
    for n in names:
        print("  ", n)
except Exception as e:
    print("\nCould not list models:", type(e).__name__, str(e)[:300])


def try_call(label, **extra):
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Say OK"}], **extra)
        print(f"{label:22} -> OK: {r.choices[0].message.content!r}")
    except APIStatusError as e:
        print(f"{label:22} -> FAILED {e.status_code}: {e.response.text[:400]}")
    except Exception as e:
        print(f"{label:22} -> ERROR {type(e).__name__}: {str(e)[:300]}")


print()
try_call("plain")
try_call("with max_tokens", max_tokens=1000)
try_call("with temperature=0", temperature=0)