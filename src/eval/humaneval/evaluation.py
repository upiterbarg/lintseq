import itertools
import numpy as np
import tqdm
import pathlib
import sys
import os

from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *

sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "eval", "humaneval"))
from execution import check_correctness


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 180.0,
    problems=None,
    problem_file: str = "",
    diff: int = False,
    malformed_penalty: int = False,
    strip_markdown: bool = True,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    if not problems:
        problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)
        completions = []

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            start = 0
            if diff:
                completion = resolve_edit_path(
                    completion, malformed_penalty=malformed_penalty
                )

            #################
            if strip_markdown:
                if "```python" in completion:
                    completion = completion[
                        completion.find("```python") + len("```python") :
                    ]
                if "```" in completion:
                    completion = completion[: completion.find("```")]
            ##################

            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k
