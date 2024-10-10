import argparse
import os
import json
import random
import torch
import vllm
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


"""
This code is largely drawn from the Tulu open-source instruction finetuning project by the Allen AI Institute.

@article{wang2023far,
  title={How far can camels go? exploring the state of instruction tuning on open resources},
  author={Wang, Yizhong and Ivison, Hamish and Dasigi, Pradeep and Hessel, Jack and Khot, Tushar and Chandu, Khyathi and Wadden, David and MacMillan, Kelsey and Smith, Noah A and Beltagy, Iz and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={74764--74786},
  year={2023}
}

@article{ivison2023camels,
  title={Camels in a changing climate: Enhancing lm adaptation with tulu 2},
  author={Ivison, Hamish and Wang, Yizhong and Pyatkin, Valentina and Lambert, Nathan and Peters, Matthew and Dasigi, Pradeep and Jang, Joel and Wadden, David and Smith, Noah A and Beltagy, Iz and others},
  journal={arXiv preprint arXiv:2311.10702},
  year={2023}
}
"""


def main(args):
    from astroid import MANAGER

    MANAGER.astroid_cache.clear()
    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prediction_save_path = os.path.join(args.save_dir, "codex_eval_predictions.jsonl")
    if os.path.exists(prediction_save_path):
        return

    test_data = list(read_problems(args.data_file).values())
    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("Number of examples:", len(test_data))

    if args.prompt_version == 0:
        if args.diff:
            prompts = [(example["prompt"] + f"\n{DIFF_TOKEN}") for example in test_data]
        else:
            prompts = [example["prompt"] for example in test_data]
    elif args.prompt_version == 1:
        prompts = []
        with open(args.data_file_hep, "r") as f:
            instructions = [json.loads(l) for l in f]
            instructions_dict = {
                x["task_id"].replace("Python", "HumanEval"): x["instruction"]
                for x in instructions
            }
        for example in test_data:
            instruction = instructions_dict[example["task_id"]]
            if args.diff:
                prompts += [(instruction + f"\n{DIFF_TOKEN}")]
            else:
                if args.use_markdown:
                    prompts += [instruction + "\n```python\n"]
                else:
                    prompts += [instruction + "\n"]

    tokenizer = load_hf_tokenizer(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        use_fast_tokenizer=not args.use_slow_tokenizer,
    )
    if args.use_vllm:
        if "llama" in args.model_name_or_path:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path
                if args.tokenizer_name_or_path
                else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                max_num_seqs=64,
            )
        else:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path
                if args.tokenizer_name_or_path
                else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
        sampling_params = vllm.SamplingParams(
            n=args.unbiased_sampling_size_n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,
            stop=[EOS_TOKEN],
            min_tokens=8,
        )

        generations = model.generate(prompts, sampling_params)
        outputs = [output.text for it in generations for output in it.outputs]
        outputs = [output for output in outputs]

    else:
        print("Loading model and tokenizer...")
        model = load_hf_lm(
            model_name_or_path=args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            # device map is determined by the number of gpus available.
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
        )

        stop_sequences = None
        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            do_sample = args.temperature != 0
            samping_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                do_sample=do_sample,  # if only pass@1 is evaluated, we do greedy decoding.
                top_p=args.top_p,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs_per_sampling_iter.append(samping_outputs)
        # regroup the outputs to match the number of test data.
        outputs = []
        for i in range(len(prompts)):
            for j in range(args.unbiased_sampling_size_n):
                outputs.append(outputs_per_sampling_iter[j][i])

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [
        (example, prompt)
        for (example, prompt) in zip(test_data, prompts)
        for _ in range(args.unbiased_sampling_size_n)
    ]
    assert len(duplicate_test_data) == len(outputs)
    predictions = [
        {
            "task_id": example["task_id"],
            "prompt": prompt,
            "completion": output,
        }
        for ((example, prompt), output) in zip(duplicate_test_data, outputs)
    ]

    write_jsonl(prediction_save_path, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--malformed_penalty",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="eval_data/HumanEval.jsonl.gz",
        help="Path to the HumanEval data file.",
    )
    parser.add_argument(
        "--data_file_hep",
        type=str,
        default="eval_data/humanevalpack.jsonl",
        help="Path to the HumanEvalPack data file.",
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/codex_eval",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n",
        type=int,
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. ",
    )
    parser.add_argument(
        "--use_markdown",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput.",
    )

    args = parser.parse_args()
    main(args)
