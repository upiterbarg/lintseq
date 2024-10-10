import argparse
import os
import json
import random
import torch
import jsonlines
from datasets import load_dataset
import sys
import pathlib
import os
import pdb

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *

sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "eval", "mbpp"))
from evaluation import compute_code_eval


def main(args):
    from astroid import MANAGER

    MANAGER.astroid_cache.clear()
    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prediction_save_path = os.path.join(args.save_dir, "mbpp_predictions")

    prediction_save_path = prediction_save_path + ".jsonl"
    if not os.path.exists(prediction_save_path):
        raise ValueError(f"{prediction_save_path} does not exist!!")

    metrics_file = os.path.join(args.save_dir, "metrics.json")
    if args.malformed_penalty:
        metrics_file = os.path.join(args.save_dir, "metrics_malpenalty.json")

    if os.path.exists(metrics_file):
        print(f"metrics have already been computed at {metrics_file}")
        return

    reader = jsonlines.open(prediction_save_path, "r")
    predictions = [prediction for prediction in reader]

    dataset = load_dataset("evalplus/mbppplus")["test"]
    tests = {example["task_id"]: example["test"] for example in dataset}

    test_predictions = []
    for prediction in predictions:
        test_predictions += [
            {**prediction, **{"test_cases": tests[prediction["task_id"]]}}
        ]

    pass_at_k_results, results = compute_code_eval(
        prediction_save_path,
        predictions=test_predictions,
        k=args.eval_pass_at_ks,
        num_workers=16,
        timeout=10,
        diff=args.diff,
        malformed_penalty=args.malformed_penalty,
        prompt_version=args.prompt_version,
    )

    print(pass_at_k_results)
    with open(metrics_file, "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/codex_eval",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
    )
    parser.add_argument(
        "--eval_pass_at_ks",
        nargs="+",
        type=int,
        default=[1],
        help="Multiple k's that we will report pass@k.",
    )
    parser.add_argument(
        "--diff",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--malformed_penalty",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    main(args)
