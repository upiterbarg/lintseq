import functools
import jsonlines
import pathlib
import sys
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from argparse import ArgumentParser
from copy import deepcopy
import time

####
REPO_NAME = "lintseq"
base_path = str(pathlib.Path().resolve())
PROJECT_PATH = base_path[: base_path.rfind(REPO_NAME) + len(REPO_NAME)]
####
sys.path.insert(0, os.path.join(PROJECT_PATH, "src"))
from utils import *

sys.path.insert(0, os.path.join(PROJECT_PATH, "src", "data_gen"))
from lintseq import *

"""Multithreaded synthetic edit sequence generation in Python with LintSeq."""


def gen_edit_paths(idx, start_i, total_samples, args, df, samples):
    """
    Batch generation of edit paths.
    Each process will generate edit paths for a chunk of the data to minimize task switching.
    """
    data = []
    for i in range(start_i, start_i + total_samples):
        index = samples[i]
        code_as_text = df[args.data_key][index]
        code_as_text = strip_chain_of_thought(code_as_text)

        for _ in range(args.num_edit_paths_per_sample):
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
                continue  # Skip errors, don't stop the whole process

            if edit_path is None:
                continue

            edit_sequence = edit_path[0][0]

            _, diff_seq = inflate_edit_path(code_as_text, edit_sequence)

            datum = {
                "edit_path": diff_seq,
                "index": int(index),
                "source_file": args.source,
                "source_instruction": df["instruction"][index],
                "source_response": df[args.data_key][index],
            }
            data.append(datum)
    return data


def main(args):
    set_seed_everywhere(args.seed)

    df = pd.read_json(args.source, lines=True)
    try:
        args.num_samples = int(args.num_samples)
        samples = np.random.choice(
            np.arange(len(df)), size=(args.num_samples,), replace=False
        )
    except:
        samples = np.arange(len(df))

    samples = np.array(samples)

    # Batch work to reduce inter-process communication
    # num_paths_per_proc = min(8, samples.shape[0] // num_proc)
    num_paths_per_proc = 8
    num_proc = 128

    gen_edit_paths_helper = functools.partial(
        gen_edit_paths, args=args, df=df, samples=samples
    )

    gen_args = []
    start_i = 0
    for i in range(0, samples.shape[0], num_paths_per_proc):
        gen_args.append(
            [i // num_paths_per_proc, i, min(num_paths_per_proc, samples.shape[0] - i)]
        )

    total_tasks = samples.shape[0] * args.num_edit_paths_per_sample

    # Single progress bar for the entire processing
    with tqdm(total=total_tasks, desc="Processing", ncols=100) as pbar:
        # Write the output file in larger batches
        with jsonlines.open(args.dest, mode="w") as writer:
            with ProcessPoolExecutor(max_workers=num_proc) as executor:
                futures = [
                    executor.submit(gen_edit_paths_helper, *args) for args in gen_args
                ]

                batch_results = []
                batch_size = (
                    num_paths_per_proc * args.num_edit_paths_per_sample
                )  # Write results in larger batches to reduce I/O frequency,

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        if results:
                            batch_results.extend(results)
                            if len(batch_results) >= batch_size:
                                writer.write_all(batch_results)
                                pbar.update(len(batch_results))
                                batch_results.clear()
                    except Exception as e:
                        print(f"Error processing future: {e}")
                    finally:
                        del future

                # Write any remaining results after all futures are processed
                if batch_results:
                    writer.write_all(batch_results)
                    pbar.update(len(batch_results))


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
        default=os.path.join(
            PROJECT_PATH, "instruct_data/merged_oss_data_raw_pyt.jsonl"
        ),
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
        default=256,
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
        default=1,
        type=int,
        help="How many edit paths should we (independently) sample per example in the dataset?",
    )

    args = parser.parse_args()

    args.dest_dir = os.path.join(PROJECT_PATH, args.dest_dir)
    os.makedirs(args.dest_dir, exist_ok=True)
    args.dest = os.path.join(
        args.dest_dir,
        f"{len([f for f in os.listdir(args.dest_dir)])}".zfill(4)
        + f"_{args.num_samples}_{args.num_edit_paths_per_sample}_{args.seed}_vec.jsonl",
    )
    print(args.dest)
    return args


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    main(args)
    print(f"Completed in {time.time() - start_time:.2f} seconds")
