# LLM-A* — Table 1 reproduction and proposed method
This code base is built on the official LLM-A* repository
(Meng et al., 2024, https://github.com/SilinMeng0510/llm-astar, MIT license, see `LICENSE`).

# SETUP
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

# COMMANDS FOR REPRODUCING TABLE 1 USING YOUR OWN API
First set the API key:
Using ChatGPT:
```bash
export OPENAI_API_KEY="INSERT_KEY"
```

Using Gemini:
```bash
export GEMINI_API_KEY="INSERT_KEY"
```

For recreating table 1 using your own model:
```bash
python table1_benchmark.py --configs gpt:standard,gpt:cot,gpt:repe --outdir INSERT_NAME_FOR_FILE --batch 1
```
^^ NOTE: This only uses the first three configs, if llama3 also is to be used then don't write --configs, then it will run all configurations. However in that case a token for llama must also be given: `export HF_TOKEN="INSERT_KEY"`.

To recreate the table using our method (bidirectional A* and the others):
```bash
python table1_bi_benchmark.py --configs gpt:standard --outdir INSERT_NAME_FOR_FILE
```

For recreating the table where dynamic weighting was used instead of static weighting use these three commands, where the --results is the table you want to test it on, so if you want to use presaved ones then use either results_gemini_run_1 or results_gemini_run_2:
```bash
python dwa_baseline.py
python merge_dwa_fallback.py --results results_gemini_run_1 --dwa results_dwa/dwa_refresh_w2_d0.99.csv
python split_by_waypoints.py --results results_gemini_run_1
```

# COMMANDS FOR REPRODUCING TABLE 1 WITH OUR RESULTS (USING GEMINI)
The per-case outputs of our two runs are saved in `results_gemini_run_1/` (run A) and
`results_gemini_run_2/` (run B), as `gpt_standard.csv`, `gpt_cot.csv`, `gpt_repe.csv`.
The file prefix `gpt` is the name of the model slot in the original code; the model actually
used is Gemini.

```bash
python table1_benchmark.py --table-only --outdir results_gemini_run_1 --configs gpt:standard,gpt:cot,gpt:repe
```

Expected output for run A (geometric means over 1,000 cases, intervals were computed separately):

| Prompt   | OR (%) [95% CI]      | SR (%) [95% CI]      | RL (%) | VPR (%) |
|----------|----------------------|----------------------|--------|---------|
| standard | 49.16 [45.2; 53.4]   | 68.93 [66.3; 71.6]   | 102.76 | 100     |
| CoT      | 51.98 [47.6; 56.5]   | 70.96 [68.0; 74.0]   | 102.44 | 100     |
| RePE     | 49.98 [45.8; 54.5]   | 69.60 [66.6; 72.8]   | 102.68 | 100     |

Run B (`--outdir results_gemini_run_2`) gives OR 47.55 / 49.91 / 48.88,
SR 67.56 / 69.50 / 68.63 and RL 102.58 / 102.46 / 102.48 for standard / CoT / RePE (VPR 100).

OR = operation ratio, SR = storage ratio, RL = relative path length, VPR = valid path ratio
(all relative to A*, as in Table 1 of the paper).

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
- SSL verification is on. Only behind a proxy that intercepts HTTPS, set
  `LLMASTAR_INSECURE_SSL=1`.
- Without a key, the code stops with a message listing the variables to set.

## Contents

```
llmastar/                 A*, LLM-A* and model wrappers (official code, adapted)
dataset/environment_50_30.json   the 100 public maps (10 start-goal pairs each = 1,000 cases)
dwa_baseline.py           dynamic weighted A* baseline (initial weight 2, decay 0.99) on all 1,000 cases;
                          writes results_dwa/dwa_refresh_w2_d0.99.csv
merge_dwa_fallback.py     rebuilds the LLM-A* table with dynamic weighting instead of the static weight-2
                          fallback in the cases where the LLM gave no waypoints; needs no API key
split_by_waypoints.py     splits the LLM-A* results into cases with 0 LLM waypoints and cases with 1 or
                          more
results_gemini_run_1/     run A: per-case results of a full 1,000-case run (Few-Shot, CoT, RePE)
results_gemini_run_2/     run B: second full run
results_bi/               saved bidirectional benchmark results (1,000 cases) and summary CSV
requirements.txt          Python dependencies
table1_benchmark.py       runs A* and LLM-A* on the dataset and writes one CSV per prompt
table1_bi_benchmark.py    evaluates A*, Bi A*, and Optimal Bidirectional LLM-A* ablations
```

## Proposed algorithm

Our proposed method is **Optimal Bidirectional LLM-A\*** (`BiLLMAStarFast`), which guarantees
grid-optimal paths (`RL = 100.00%`) regardless of LLM waypoint quality while accelerating search
from both start and goal frontiers.

### How It Works
1. **Bidirectional Frontier Search (`optimal_core.py`):** Runs simultaneous forward (start-to-goal)
   and backward (goal-to-start) A* searches using an admissible octile heuristic and prunes
   branches once `min_f_fwd + min_f_bwd >= best_path_cost`.
2. **LLM Incumbent Upper Bound (`use_incumbent=True`):** Connects consecutive LLM waypoints via
   bounded A* hops (`hop_cap_factor=20`) to obtain a valid initial feasible path cost `mu` early,
   immediately pruning states with `f >= mu`.
3. **LLM Corridor Tie-Breaking Guide (`use_guide=True`):** Builds a bounded BFS distance field
   (`guide_radius=8`) around the LLM waypoint path and uses it strictly as a secondary tie-breaker
   among equal-`f` nodes (`guide_first=True`), preserving admissibility and optimality.

### Evaluate Against A* and LLM-A*

```bash
# Rebuild the saved 1,000-case bidirectional comparison table WITHOUT an API key:
python table1_bi_benchmark.py --outdir INSERT_NAME_FOR_FOLDER --baseline astar --table-only

# Run live evaluation on the dataset (requires an API key; caches LLM plans in results_bi/llm_cache.json):
python table1_bi_benchmark.py --outdir INSERT_NAME_FOR_FOLDER --configs gpt:standard --baseline astar
```

### Citation

```
@article{meng2024llm,
  title={LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning},
  author={Meng, Silin and Wang, Yiwei and Yang, Cheng-Fu and Peng, Nanyun and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2407.02511},
  year={2024}
}
```

## License
MIT. Check `LICENSE`.