import functools
import jsonlines
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import pdb
import random
import shutil
import sys
import tempfile

from argparse import ArgumentParser
from tqdm import tqdm

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *

"""Multithreaded random edit sequence generation -- LintSeq linter ablation."""


def random_chunked_trajectory(
    code_as_text, ignore_imports=False, ignore_comments=False, ignore_global_defs=False
):
    lines = code_as_text.split("\n")
    candidate_lines = [i for i in range(len(lines))]

    edit_path = []

    while len(candidate_lines) > 0:
        if len(candidate_lines) == 1:
            chunk = candidate_lines
        else:
            chunk_size = np.random.randint(1, high=len(candidate_lines))
            chunk = random.sample(candidate_lines, chunk_size)
        edit_path += [chunk]
        candidate_lines = [i for i in candidate_lines if not i in chunk]

    return edit_path


def gen_edit_paths(idx, start_i, total_samples, args, df, samples):
    data = []
    with tqdm(
        total=(total_samples * args.num_edit_paths_per_sample),
        position=idx,
        desc=str(os.getpid()),
    ) as pbar:
        for i in range(start_i, start_i + total_samples):
            index = samples[i]
            code_as_text = df[args.data_key][index]
            code_as_text = strip_chain_of_thought(code_as_text)

            for j in range(args.num_edit_paths_per_sample):
                edit_sequence = random_chunked_trajectory(code_as_text)
                _, diff_seq = inflate_edit_path(code_as_text, edit_sequence)

                datum = {
                    "edit_path": diff_seq,
                    "index": int(index),
                    "source_file": args.source,
                    "source_instruction": df["instruction"][index],
                    "source_response": df[args.data_key][index],
                }
                data += [datum]
                pbar.update(1)
    return data


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
        + f"_n{args.num_samples}_s{args.num_edit_paths_per_sample}_rs{args.seed}"
        + "_randomchunked.jsonl",
    )

    data = []

    ### MULTI-THREAD
    num_proc = min(multiprocessing.cpu_count(), args.num_workers, samples.shape[0])
    num_paths_per_proc = samples.shape[0] // num_proc + 1

    gen_edit_paths_helper = functools.partial(
        gen_edit_paths, args=args, df=df, samples=samples
    )

    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(num_proc - 1)):
        gen_args += [[j, start_i, num_paths_per_proc]]
        start_i += num_paths_per_proc
    if samples.shape[0] - start_i > 0:
        gen_args += [[num_proc - 1, start_i, samples.shape[0] - start_i]]

    # Run pool
    pool = multiprocessing.Pool(num_proc)
    runs = [
        pool.apply_async(gen_edit_paths_helper, args=gen_args[k])
        for k in range(num_proc)
        if len(gen_args) > k
    ]
    results = [p.get() for p in runs]
    data = []
    for data_vec in results:
        data += data_vec

    with jsonlines.open(args.dest, "w") as writer:
        writer.write_all(data)


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
        default=os.path.join(PROJECT_PATH, "instruct_data/merged_oss_data_raw_pyt.jsonl"),
        type=str,
        help="Path to source JSONLines file.",
    )
    parser.add_argument(
        "--dest_dir",
        default="instruct_data/gen",
        type=str,
        help="""Destination directory for synthetically generated data.""",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        default="all",
        help="""Number of samples to process. If passed as an integer, subsamples the dataset 
        without replacement. Otherwise, processes all data in the source file.""",
    )
    parser.add_argument(
        "-c",
        "--num_workers",
        default=8,
        type=int,
        help="Number of parallel workers to use during synthetic data generation.",
    )
    parser.add_argument(
        "--data_key",
        default="response",
        help="Name of example data field in the source dataset.",
    )
    parser.add_argument(
        "--num_edit_paths_per_sample",
        default=5,
        type=int,
        help="How many edit paths should we (independently) sample per example in the dataset?",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
