# Statistical analysis scripts (Table 1 replication)

Requirements: pip install numpy pandas scipy

1. Put the 1000-case CSVs (gpt_standard.csv, gpt_cot.csv, gpt_repe.csv) in results_gemini_1000/
   (recoverable with: git show 548b661:results_gemini/gpt_standard.csv > results_gemini_1000/gpt_standard.csv, etc.)
2. python stats_tools/llm_astar_stats.py --dir results_gemini_1000 --out stats_out
3. python stats_tools/llm_astar_justification.py --run1 results_gemini_1000 --run2 results_gemini --out just_out

Outputs are CSV tables used in the report (confidence intervals, tests vs the paper, run-to-run noise).
