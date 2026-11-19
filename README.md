# LLM-A* — Table 1 reproduction (Gemini backbone) and proposed method

This code base is built on the official LLM-A* repository
(Meng et al., 2024, https://github.com/SilinMeng0510/llm-astar, MIT license, see `LICENSE`).
Modifications made for this project: a configurable OpenAI-compatible backend, used with Gemini and GPT-4o mini (`llmastar/model/chatgpt_gemini.py`),
a resumable Table 1 benchmark (`table1_benchmark.py`), offline statistics (`table1_stats.py`,
`stats_tools/`), and the saved results of our runs (`results_gemini_1000/`, `results_gemini/`).

## 1. Contents

```
llmastar/                 A*, LLM-A* and model wrappers (official code, adapted)
dataset/environment_50_30.json   the 100 public maps (10 start-goal pairs each = 1,000 cases)
table1_benchmark.py       runs A* and LLM-A* on the dataset and writes one CSV per prompt
table1_stats.py           recomputes Table 1 + standard deviations / confidence intervals from the CSVs
results_gemini_1000/      run A: per-case results of a full 1,000-case run (standard, cot, repe)
results_gemini/           run B: second full run (1,000 cases per prompt), same model and settings
stats_tools/              analysis scripts used in the report (two-run analysis, no-waypoint ablation)
ablation_out/, two_run_out/   output tables of these scripts (CSV)
requirements.txt          Python dependencies
```

## 2. Setup (Python 3.11, no GPU)

Developed and run on Windows with Python 3.11; the offline commands below were also checked
on Linux (Ubuntu, Python 3.12) in a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No API key is stored in this repository. Keys are only read from environment variables.

## 3. Reproduce Table 1 WITHOUT an API key

The per-case outputs of our two runs are saved in `results_gemini_1000/` (run A) and
`results_gemini/` (run B), as `gpt_standard.csv`, `gpt_cot.csv`, `gpt_repe.csv`.
The file prefix `gpt` is the name of the model slot in the original code; the model actually
used is Gemini, see Section 4.

```bash
python table1_benchmark.py --table-only --outdir results_gemini_1000 --configs gpt:standard,gpt:cot,gpt:repe
python table1_stats.py --dir results_gemini_1000 --out table1_with_stats.csv
```

Expected output for run A (geometric means over 1,000 cases, 95% cluster-bootstrap intervals over the 100 maps):

| Prompt   | OR (%) [95% CI]      | SR (%) [95% CI]      | RL (%) | VPR (%) |
|----------|----------------------|----------------------|--------|---------|
| standard | 49.16 [45.2; 53.4]   | 68.93 [66.3; 71.6]   | 102.76 | 100     |
| CoT      | 51.98 [47.6; 56.5]   | 70.96 [68.0; 74.0]   | 102.44 | 100     |
| RePE     | 49.98 [45.8; 54.5]   | 69.60 [66.6; 72.8]   | 102.68 | 100     |

Run B (`--outdir results_gemini` / `--dir results_gemini`) gives OR 47.55 / 49.91 / 48.88,
SR 67.56 / 69.50 / 68.63 and RL 102.58 / 102.46 / 102.48 for standard / CoT / RePE (VPR 100).

OR = operation ratio, SR = storage ratio, RL = relative path length, VPR = valid path ratio
(all relative to A*, as in Table 1 of the paper). `table1_stats.py` also prints the standard
deviation of the per-case log-ratios and the bootstrap standard error.

### Additional statistics used in the report (no API key needed)

```bash
python stats_tools/ablation_no_waypoints.py --ref results_gemini_1000 --out ablation_out
python stats_tools/two_run_analysis.py --run-a results_gemini_1000 --run-b results_gemini --out two_run_out --ablation ablation_out/no_waypoints.csv
```

`results_gemini_1000/` (run A) and `results_gemini/` (run B) are two complete runs of the same
1,000 cases with the same model and settings. `ablation_no_waypoints.py` runs the LLM-A* search
with an empty waypoint list (no LLM call, about 1-3 minutes) and writes `ablation_out/`.
`two_run_analysis.py` writes `two_run_out/`: Table 1 for each run and pooled, run-to-run
variability, prompt contrasts, comparison with the paper, and the comparison of the LLM
waypoints with the empty-waypoint search.

## 4. Re-run the LLM experiments (requires a key)

The model wrapper (`llmastar/model/chatgpt_gemini.py`) works with any OpenAI-compatible endpoint
and is configured only through environment variables. Never write a key in a file.

| Variable | Meaning | Default |
|----------|---------|---------|
| `LLM_API_KEY` (or `OPENAI_API_KEY`) | API key | none (required) |
| `LLM_BASE_URL` (or `OPENAI_BASE_URL`) | endpoint | the OpenAI endpoint |
| `LLM_MODEL` | model name | `gpt-4o-mini` |

```bash
# GPT-4o mini (OpenAI)
export LLM_API_KEY=...            # Windows PowerShell: $env:LLM_API_KEY="..."
python check_llm.py               # one test call; prints the model, the endpoint and the answer
python table1_benchmark.py --outdir my_results --configs gpt:standard,gpt:cot,gpt:repe
python table1_stats.py --dir my_results
```

Gemini (used for our Table 1 runs): set only `GEMINI_API_KEY` instead. If no `LLM_API_KEY` /
`OPENAI_API_KEY` is set, the code then uses the Gemini OpenAI-compatible endpoint and
`gemini-3.5-flash-lite` by default (`LLM_BASE_URL` and `LLM_MODEL` can still override them).

Settings used for our runs:

- Model: `gemini-3.5-flash-lite`, called through Gemini's OpenAI-compatible endpoint,
  temperature 0.
- Output-token budget per waypoint call: `--max-tokens` of `table1_benchmark.py`
  (default 8000; it overrides the value 1000 written in `llm_a_star.py`).
- `table1_benchmark.py` is resumable: cases already present in the output CSVs are skipped,
  and it stops cleanly when the API quota is reached (run the same command again later).
  `--reverse` processes the environments from the last one to the first.
- SSL verification is on. Only behind a proxy that intercepts HTTPS, set
  `LLMASTAR_INSECURE_SSL=1`.
- Without a key, the code stops with a message listing the variables to set.

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