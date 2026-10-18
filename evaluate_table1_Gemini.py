import copy
import csv
import json
import os
import numpy as np
from llmastar.pather import AStar, LLMAStar

# Set your Groq API key (or retrieve from system environment)
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or GROQ_API_KEY

def evaluate_and_save_csv(
    dataset_path="dataset/environment_50_30.json",
    output_csv="table1_results.csv",
    llm_model="gpt",
    prompt_style="standard"
):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    with open(dataset_path, "r") as f:
        environments = json.load(f)

    # Metric collections
    op_ratios = []
    storage_ratios = []
    valid_paths = 0
    total_test_cases = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    # Open CSV file to record per-query instance results
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        # Header Row for Individual Test Cases
        writer.writerow([
            "env_id", 
            "start", 
            "goal", 
            "astar_ops", 
            "astar_storage", 
            "astar_length", 
            "llm_ops", 
            "llm_storage", 
            "llm_length", 
            "op_ratio", 
            "storage_ratio", 
            "is_valid"
        ])

        print(f"--- Running Benchmark for {llm_model.upper()} ({prompt_style}) ---")

        for env in environments:
            env_id = env.get("id")
            for sg in env["start_goal"]:
                total_test_cases += 1
                start, goal = list(sg[0]), list(sg[1])

                # Build fresh query dictionary (use copy to prevent mutation)
                query = {
                    "start": copy.deepcopy(start),
                    "goal": copy.deepcopy(goal),
                    "size": [env["range_x"][1], env["range_y"][1]],
                    "horizontal_barriers": copy.deepcopy(env["horizontal_barriers"]),
                    "vertical_barriers": copy.deepcopy(env["vertical_barriers"]),
                    "range_x": copy.deepcopy(env["range_x"]),
                    "range_y": copy.deepcopy(env["range_y"])
                }

                # 1. Standard A* Baseline
                try:
                    astar_solver = AStar()
                    astar_result = astar_solver.searching(query=copy.deepcopy(query))
                    
                    astar_stats = astar_result[2] if isinstance(astar_result, tuple) and len(astar_result) >= 3 else astar_result
                    a_ops = astar_stats.get("operation", astar_stats.get("operations", 0)) if isinstance(astar_stats, dict) else 0
                    a_storage = astar_stats.get("storage", astar_stats.get("max_storage", 0)) if isinstance(astar_stats, dict) else 0
                    a_len = astar_stats.get("length", 0) if isinstance(astar_stats, dict) else 0
                except Exception as e:
                    print(f"A* Baseline Error on Env {env_id}, Start {start}: {e}")
                    a_ops, a_storage, a_len = 0, 0, 0

                # 2. LLM-A* Pathfinder
                try:
                    llm_solver = LLMAStar(llm=llm_model, prompt=prompt_style)
                    llm_result = llm_solver.searching(query=copy.deepcopy(query))
                    
                    if isinstance(llm_result, tuple):
                        llm_path = llm_result[0]
                        llm_stats = llm_result[2] if len(llm_result) >= 3 else (llm_result[1] if isinstance(llm_result[1], dict) else {})
                    else:
                        llm_path = llm_result
                        llm_stats = {}

                    l_ops = llm_stats.get("operation", llm_stats.get("operations", 0)) if isinstance(llm_stats, dict) else 0
                    l_storage = llm_stats.get("storage", llm_stats.get("max_storage", 0)) if isinstance(llm_stats, dict) else 0
                    l_len = llm_stats.get("length", 0) if isinstance(llm_stats, dict) else 0

                    # Validate path endpoint safely
                    is_valid = False
                    if llm_path and isinstance(llm_path, list) and len(llm_path) > 0:
                        last_pt = list(llm_path[-1])
                        is_valid = (last_pt == goal)

                    if is_valid:
                        valid_paths += 1

                    # Compute ratios relative to baseline
                    op_ratio = l_ops / a_ops if a_ops > 0 else 0.0
                    storage_ratio = l_storage / a_storage if a_storage > 0 else 0.0

                    if a_ops > 0:
                        op_ratios.append(op_ratio)
                        storage_ratios.append(storage_ratio)

                    # Log row to CSV
                    writer.writerow([
                        env_id, start, goal,
                        a_ops, a_storage, a_len,
                        l_ops, l_storage, l_len,
                        f"{op_ratio:.4f}", f"{storage_ratio:.4f}", is_valid
                    ])

                    print(f"Env {env_id} | Start: {start} -> Goal: {goal} | Valid: {is_valid} | Op Ratio: {op_ratio:.4f}")

                except Exception as e:
                    print(f"LLM-A* Error executing Env {env_id}, Start {start}: {e}")
                    writer.writerow([
                        env_id, start, goal,
                        a_ops, a_storage, a_len,
                        0, 0, 0,
                        0.0, 0.0, False
                    ])

        # Calculate Overall Summary Metrics
        success_rate = (valid_paths / total_test_cases * 100) if total_test_cases > 0 else 0.0
        avg_op_ratio = np.mean(op_ratios) if op_ratios else 0.0
        avg_storage_ratio = np.mean(storage_ratios) if storage_ratios else 0.0

        # Append Summary Block at the end of the CSV
        writer.writerow([])
        writer.writerow(["=== TABLE 1 SUMMARY RESULTS ==="])
        writer.writerow(["Model / Prompt Style", f"{llm_model} ({prompt_style})"])
        writer.writerow(["Total Test Cases", total_test_cases])
        writer.writerow(["Valid Path Ratio (%)", f"{success_rate:.2f}%"])
        writer.writerow(["Mean Operation Ratio", f"{avg_op_ratio:.4f}"])
        writer.writerow(["Mean Storage Ratio", f"{avg_storage_ratio:.4f}"])

    print("\n================== TABLE 1 SUMMARY ==================")
    print(f"Results saved to    : {output_csv}")
    print(f"Total Test Cases    : {total_test_cases}")
    print(f"Valid Path Ratio    : {success_rate:.2f}%")
    print(f"Operation Ratio     : {avg_op_ratio:.4f}")
    print(f"Storage Ratio       : {avg_storage_ratio:.4f}")
    print("=====================================================")

if __name__ == "__main__":
    evaluate_and_save_csv()