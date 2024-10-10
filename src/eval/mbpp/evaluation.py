import itertools
import os
import tqdm
import pathlib
import sys

import numpy as np

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

sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "eval", "mbpp"))
from execution import check_correctness

# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/custom_metrics/code_eval.py#L129

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" metric executes untrusted model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).

Once you have read this disclaimer and taken appropriate precautions,
set the environment variable HF_ALLOW_CODE_EVAL="1". Within Python you can to this
with:

>>> import os
>>> os.environ["HF_ALLOW_CODE_EVAL"] = "1"

################################################################################\
"""


def compute_code_eval(
    sample_file: str,
    predictions: List,
    k: List[int] = [1, 10, 100],
    num_workers: int = 4,
    timeout: float = 3.0,
    diff: int = False,
    malformed_penalty: int = False,
    strip_markdown: bool = True,
    prompt_version: int = 0,
):
    """Returns the scores"""

    if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1":
        raise ValueError(_WARNING)

    if os.name == "nt":
        raise NotImplementedError("This metric is currently not supported on Windows.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for sample in predictions:
            completion = sample["completion"]
            if strip_markdown:
                if "```python" in completion:
                    completion = completion[
                        completion.find("```python") + len("```python") :
                    ]
                if "python" in completion:
                    completion = completion[completion.find("python") + len("python") :]
                if "```" in completion:
                    completion = completion[: completion.find("```")]

            if diff:
                completion = resolve_edit_path(
                    completion,
                    malformed_penalty=malformed_penalty,
                )
                test_program = completion + "\n" + sample["test_cases"]
            elif prompt_version == 0:
                prompt = sample["prompt"]
                prompt = prompt[prompt.find("def") :]

                test_program = prompt + completion + "\n" + sample["test_cases"]
            elif prompt_version == 1:
                test_program = completion + "\n" + sample["test_cases"]

            args = (
                test_program,
                timeout,
                sample["task_id"],
                completion_id[sample["task_id"]],
            )

            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[sample["task_id"]] += 1
            n_samples += 1

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    if not isinstance(ks, (list, tuple)):
        ks = [ks]
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        if "." in sample_file:
            sf = sample_file
        else:
            sf += ".jsonl"
        for sample in stream_jsonl(sf):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    if "." in sample_file:
        sf = sample_file[: sample_file.find(".")]
    else:
        sf = sample_file

    out_file = sf + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k, results
