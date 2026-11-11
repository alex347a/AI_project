# LLM-A* — Table 1 reproduction (Gemini backbone) and proposed method

This code base is built on the official LLM-A* repository
(Meng et al., 2024, https://github.com/SilinMeng0510/llm-astar, MIT license, see `LICENSE`).
Modifications made for this project: a Gemini backend (`llmastar/model/chatgpt_gemini.py`),
a resumable Table 1 benchmark (`table1_benchmark.py`), offline statistics (`table1_stats.py`),
and the saved results of our runs (`results_gemini_1000/`).

## 1. Contents

```
llmastar/                 A*, LLM-A* and model wrappers (official code, adapted)
dataset/environment_50_30.json   the 100 public maps (10 start-goal pairs each = 1,000 cases)
table1_benchmark.py       runs A* and LLM-A* on the dataset and writes one CSV per prompt
table1_stats.py           recomputes Table 1 + standard deviations / confidence intervals from the CSVs
results_gemini_1000/      per-case results of our full 1,000-case run (standard, cot, repe)
requirements.txt          Python dependencies
```

## 2. Setup (tested layout: Ubuntu 22.04, Python 3.11, no GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is stored in this repository. Keys are only read from environment variables.

## 3. Reproduce Table 1 WITHOUT an API key

The per-case outputs of our run are saved in `results_gemini_1000/`
(`gpt_standard.csv`, `gpt_cot.csv`, `gpt_repe.csv`; the file prefix `gpt` is the
name of the model slot in the original code, the model actually used is Gemini, see Section 4).

```bash
python table1_benchmark.py --table-only --outdir results_gemini_1000 --configs gpt:standard,gpt:cot,gpt:repe
python table1_stats.py --dir results_gemini_1000 --out table1_with_stats.csv
```

Expected output (geometric means over 1,000 cases, 95% cluster-bootstrap intervals over the 100 maps):

| Prompt   | OR (%) [95% CI]      | SR (%) [95% CI]      | RL (%) | VPR (%) |
|----------|----------------------|----------------------|--------|---------|
| standard | 49.16 [45.2; 53.4]   | 68.93 [66.3; 71.6]   | 102.76 | 100     |
| CoT      | 51.98 [47.6; 56.5]   | 70.96 [68.0; 74.0]   | 102.44 | 100     |
| RePE     | 49.98 [45.8; 54.5]   | 69.60 [66.6; 72.8]   | 102.68 | 100     |

OR = operation ratio, SR = storage ratio, RL = relative path length, VPR = valid path ratio
(all relative to A*, as in Table 1 of the paper). `table1_stats.py` also prints the standard
deviation of the per-case log-ratios and the bootstrap standard error.

## 4. Re-run the LLM experiments (requires a key)

```bash
export GEMINI_API_KEY=...        # Windows PowerShell: $env:GEMINI_API_KEY="..."
python table1_benchmark.py --outdir my_results --configs gpt:standard,gpt:cot,gpt:repe
python table1_stats.py --dir my_results
```

Settings used for our run:

- Model: `gemini-3.5-flash-lite`, called through Gemini's OpenAI-compatible endpoint
  (`llmastar/model/chatgpt_gemini.py`), temperature 0, one run per case.
- Output-token budget per call: `--max-tokens` (default 8000).
- `table1_benchmark.py` is resumable: cases already present in the output CSVs are skipped.
- Without a key, the code stops with the message `GEMINI_API_KEY is not set`.

## 5. Using the planners on a new map (input / output format)

The input format is the one of the original repository:

```python
from llmastar.pather import AStar, LLMAStar

query = {"start": [5, 5], "goal": [27, 15], "size": [51, 31],
         "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],   # [y, x_start, x_end]
         "vertical_barriers": [[25, 10, 22]],                  # [x, y_start, y_end]
         "range_x": [0, 51], "range_y": [0, 31]}

astar = AStar().searching(query=query, filepath='astar.png')
llm = LLMAStar(llm='gpt', prompt='standard').searching(query=query, filepath='llm.png')
# both return a dict: {"operation", "storage", "length", "llm_output" (LLM-A* only)}
```

`prompt` is one of `standard`, `cot`, `repe`.

## 6. Proposed algorithm

> TO COMPLETE BEFORE SUBMISSION: describe the proposed method, the file(s) that implement it,
> the exact command to run it and the exact command to evaluate it against A* and LLM-A*.
> It must be the method described in the report.

## 7. Notes

- The LLaMA3 backend of the original code needs a GPU and `torch`/`transformers`;
  it is optional and not used in our experiments (install them yourself if needed).
- Results on the public maps are not the final evaluation (hidden maps are used by the teaching staff).

## 8. Citation

```
@article{meng2024llm,
  title={LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning},
  author={Meng, Silin and Wang, Yiwei and Yang, Cheng-Fu and Peng, Nanyun and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2407.02511},
  year={2024}
}
```

## 9. License

MIT. Check `LICENSE`.