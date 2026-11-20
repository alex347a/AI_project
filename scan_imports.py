"""
List every third-party module imported by the .py files in this repo, so requirements.txt can be checked.

Run from the repo root (PowerShell or Linux):
    python scan_imports.py
It reads the code only (nothing is imported or executed) and prints, for each third-party module,
the pip package it usually comes from and the files that use it.
"""
import ast
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", "site-packages", "build", "dist"}
# import name -> pip name, where they differ or are worth stating
PIP_NAME = {"dotenv": "python-dotenv", "PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
            "yaml": "pyyaml", "groq": "groq", "openai": "openai", "httpx": "httpx", "numpy": "numpy",
            "matplotlib": "matplotlib", "shapely": "shapely", "inquirer": "inquirer", "requests": "requests",
            "ollama": "ollama", "torch": "torch", "transformers": "transformers", "pandas": "pandas",
            "scipy": "scipy", "tqdm": "tqdm"}

local, files = set(), []
for d, dirs, names in os.walk(ROOT):
    dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
    for n in names:
        if n.endswith(".py"):
            files.append(os.path.join(d, n))
            local.add(n[:-3])
    for x in dirs:
        local.add(x)

found = {}
for path in files:
    try:
        tree = ast.parse(open(path, encoding="utf-8-sig").read(), filename=path)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"[could not parse] {path}: {e}")
        continue
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m not in sys.stdlib_module_names and m not in local:
                found.setdefault(m, set()).add(os.path.relpath(path, ROOT))

if not found:
    print("No third-party imports found.")
for m in sorted(found):
    print(f"{m:14} pip: {PIP_NAME.get(m, '(look up: ' + m + ')'):16} used in: {', '.join(sorted(found[m]))}")
print(f"\nScanned {len(files)} files. Python {sys.version.split()[0]} (needs 3.10+ to run this scanner).")