import concurrent
import functools
import gc
import json
import logging
import multiprocessing
import numpy as np
import os
import time
import pandas as pd
import pathlib
import pdb
import sys
import _thread
import warnings
from typing import Optional
from queue import Queue
from threading import Thread
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from fastparquet import ParquetFile
from functools import partial
from pylint import run_pylint
from tqdm import tqdm

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *
sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "data_gen"))
from lintseq import *

warnings.filterwarnings("ignore")

"""Multithreaded synthetic edit sequence generation with LintSeq."""

class Pool:
    """Pool with maximal concurrent job load, managed via 'work' and 'out' queues."""
    def __init__(self, total_work, work, max_concurrent_jobs, max_worker: int = 32) -> None:
        self.max_workers = max_worker
        self.work_queue = Queue(max_concurrent_jobs)
        self.out_queue = Queue()
        self.is_running = True
        pbar = tqdm(total=total_work)

        def _work():
            while self.is_running:
                item = self.work_queue.get()
                result = work(*item)
                self.work_queue.task_done()
                pbar.update(1)
                self.out_queue.put(result)

        for _ in range(self.max_workers):
            Thread(target=_work).start()

    def close(self):
        self.is_running = False


def worker_fn(code_as_text, datum):
    """Worker function. Some linters (such as pylint) may have extremely high RAM usage or even
    memory leaks. Instatiating separate workers for each linting operatation limits RAM pressure."""
    try:
        edit_path = lintseq_backward_sampling_pythonic(
            code_as_text,
            children_per_round=1,
            top_k=1,
            max_population_size=1,
            max_depth=512,
            indent_bias_sampling_factor=1,
            ignore_imports=False,
            ignore_comments=True,
            ignore_global_defs=True,
            verbose=False,
            ignore_init_errors=False,
        )
    except:
        return None

    if edit_path is None:
        return None

    _, diff_seq = inflate_edit_path(code_as_text, edit_path)

    datum = {
        **datum,
        **{
            "edit_path": diff_seq,
        }
    }
    return datum


def main(args):
    set_seed_everywhere(args.seed)

    if args.source[args.source.rfind(".") :] == ".jsonl":
        df = pd.read_json(args.source, lines=True)
    else:
        raise ValueError(f"Unsupported file format {args.source[args.source.rfind(".") :]}")

    try:
        args.num_samples = int(args.num_samples)
        samples = np.random.choice(
            np.arange(len(df)), size=(args.num_samples,), replace=False
        )
    except:
        samples = np.array([i for i in range(len(df))])
        args.num_samples = len(samples)

    samples = np.array(samples)

    args.dest_dir = os.path.join(PROJECT_PATH, args.dest_dir)
    os.makedirs(args.dest_dir, exist_ok=True)
    args.dest = os.path.join(
        args.dest_dir,
        f"{len([f for f in os.listdir(args.dest_dir)])}".zfill(4)
        + f"_n{args.num_samples}_s{args.num_edit_paths_per_sample}_rs{args.seed}.jsonl",
    )
    print(args.dest)

    ### MULTI-THREAD
    total_work = len(samples) * args.num_edit_paths_per_sample
    pool = Pool(
        total_work, 
        work=worker_fn, 
        max_concurrent_jobs=total_work, 
        max_worker=args.num_workers
    )

    def worker():
        counter = 0
        with open(args.dest, "w") as outfile:
            while True:
                item = pool.out_queue.get()
                counter += 1
                if not item is None:
                    pitem = json.dumps(item) + "\n"
                    outfile.write(pitem)
                pool.out_queue.task_done()
                if counter == total_work:
                    pool.close()
                    os._exit(0)
                    return

    post_process_thread = Thread(target=worker)
    post_process_thread.start()

    for index in samples:
        code_as_text = df[args.data_key][index]
        code_as_text = strip_chain_of_thought(code_as_text)

        for j in range(args.num_edit_paths_per_sample):
            datum = {
                "index": int(index),
                "source_file": args.source,
                "source_instruction": df["instruction"][index],
                "source_response": df[args.data_key][index],
            }

            wargs = (code_as_text, datum,)
            pool.work_queue.put(wargs)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed", 
        default=1, 
        type=int,
        help="Random seed used during synthetic data generation.",
    )
    parser.add_argument(
        "--source",
        default="/data/projects/editregress/instruct_data/merged_oss_data_raw_pyt.jsonl",
        type=str,
        help="Path to source JSONLines file."
    )
    parser.add_argument(
        "--dest_dir", 
        default="instruct_data/gen", 
        ype=str, 
        help="""Destination directory for synthetically generated data."""
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default="all",
        help="""Number of samples to process. If passed as an integer, subsamples the dataset 
        without replacement. Otherwise, processes all data in the source file."""
    )
    parser.add_argument(
        "-c", 
        "--num_workers", 
        default=256, 
        type=int,
        help="Number of parallel workers to use during synthetic data generation."
    )
    parser.add_argument(
        "--data_key", 
        default="response",
        help="Name of example data field in the source dataset."
    )
    parser.add_argument(
        "--num_edit_paths_per_sample", 
        default=5, 
        type=int,
        help="How many edit paths should we (independently) sample per example in the dataset?"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
