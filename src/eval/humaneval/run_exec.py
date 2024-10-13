import argparse
import os
import json
import random
import torch
import jsonlines
import sys
import pathlib
import os

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *

sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "eval", "humaneval"))
from evaluation import evaluate_functional_correctness


def main(args):
    from astroid import MANAGER

    MANAGER.astroid_cache.clear()
    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prediction_save_path = os.path.join(args.save_dir, "codex_eval_predictions.jsonl")
    if not os.path.exists(prediction_save_path):
        raise ValueError(f"{prediction_save_path} does not exist!!")

    metrics_file = os.path.join(args.save_dir, "metrics.json")
    if args.malformed_penalty:
        metrics_file = os.path.join(args.save_dir, "metrics_malpenalty.json")

    print(metrics_file)
    if not os.path.exists(metrics_file):
        test_data = list(read_problems(args.data_file).values())
        if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)
        print("Number of examples:", len(test_data))

        pass_at_k_results = evaluate_functional_correctness(
            sample_file=prediction_save_path,
            k=args.eval_pass_at_ks,
            problems={example["task_id"]: example for example in test_data},
            n_workers=64,
            diff=args.diff,
            malformed_penalty=args.malformed_penalty,
            timeout=10,
        )

        print(pass_at_k_results)

        with open(metrics_file, "w") as fout:
            json.dump(pass_at_k_results, fout)
    else:
        print(f"metrics have already been computed at {metrics_file}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--data_file",
        type=str,
        default="data/codex_eval/HumanEval.jsonl.gz",
        help="Path to the HumanEval data file.",
    )
    parser.add_argument(
        "--data_file_hep",
        type=str,
        default="data/eval/humaneval/humanevalpack.jsonl",
        help="Path to the HumanEvalPack data file.",
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/codex_eval",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--eval_pass_at_ks",
        nargs="+",
        type=int,
        default=[1],
        help="Multiple k's that we will report pass@k.",
    )
    args = parser.parse_args()
    main(args)
