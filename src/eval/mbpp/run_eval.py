import argparse
import os
import json
import random
import torch
import vllm
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


def main(args):
    from astroid import MANAGER

    MANAGER.astroid_cache.clear()
    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    prediction_save_path = os.path.join(args.save_dir, "mbpp_predictions")
    prediction_save_path = prediction_save_path + ".jsonl"
    if os.path.exists(prediction_save_path):
        return

    dataset = load_dataset("evalplus/mbppplus")["test"]
    dataset.shuffle(seed=42)

    # Always head-out first 100 examples
    if args.max_num_examples is None:
        args.max_num_examples = len(dataset) - 100
    if args.max_num_examples > len(dataset) - 100:
        Warning(
            "The number of examples is larger than the test set size. Will use the maximum number of examples."
        )
        args.max_num_examples = len(dataset) - 100
    test_data = dataset.select(
        range(100, min(100 + args.max_num_examples, len(dataset)))
    )
    print("Number of examples:", len(test_data))

    if args.prompt_version == 0:
        if args.diff:
            prompts = [
                example["prompt"] + example["code"].split(":")[0] + f"\n{DIFF_TOKEN}"
                for example in test_data
            ]
        elif args.use_markdown:
            prompts = [
                example["prompt"] + "```python\n" + example["code"].split(":")[0]
                for example in test_data
            ]
        else:
            prompts = [
                example["prompt"] + example["code"].split(":")[0]
                for example in test_data
            ]
    elif args.prompt_version == 1:
        prompts = []
        for example in test_data:
            fdef = example["code"].lstrip().split("\n")
            fdef = [line for line in fdef if "def" in line][0]
            fdef = fdef.rstrip().rstrip(":").lstrip("def").lstrip()
            if args.diff:
                prompts += [
                    example["prompt"].replace("function", f"function `{fdef}`")
                    + f"\n{DIFF_TOKEN}"
                ]

            elif args.use_markdown:
                prompts += [
                    example["prompt"].replace("function", f"function `{fdef}`")
                    + "\n```python\n"
                ]
            else:
                prompts += [
                    example["prompt"].replace("function", f"function `{fdef}`") + "\n"
                ]

    tokenizer = load_hf_tokenizer(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path,
        use_fast_tokenizer=not args.use_slow_tokenizer,
    )

    if args.use_stop_sequences:
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    else:
        stop_sequences = None

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
            max_tokens=512,
            stop=stop_sequences,
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
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM

        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(
                "Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                    model.config.max_position_embeddings
                )
            )

        if args.use_chat_format:
            prompts = [
                apply_chat_format(tokenizer, inst, suffix) for (inst, suffix) in prompts
            ]

        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            do_sample = args.temperature != 0
            import pdb

            if not stop_sequences is None:
                # Because many tokenizers will treat the word after space differently from the original word alone,
                # to be consistent, we add a space before tokenization and remove it after tokenization.
                stop_sequences = [
                    tokenizer.encode(" " + x, add_special_tokens=False)[1:]
                    for x in stop_sequences
                ]

            sampling_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                do_sample=do_sample,  # if only pass@1 is evaluated, we do greedy decoding.
                stop_id_sequences=stop_sequences,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs_per_sampling_iter.append(sampling_outputs)
        # regroup the outputs to match the number of test data.
        outputs = []
        for i in range(len(prompts)):
            for j in range(args.unbiased_sampling_size_n):
                outputs.append(outputs_per_sampling_iter[j][i])

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [
        example for example in test_data for _ in range(args.unbiased_sampling_size_n)
    ]
    duplicate_prompts = [
        prompt for prompt in prompts for _ in range(args.unbiased_sampling_size_n)
    ]

    predictions_noresult = [
        {
            "task_id": example["task_id"],
            "prompt": prompt,
            "completion": output,
        }
        for example, prompt, output in zip(
            duplicate_test_data, duplicate_prompts, outputs
        )
    ]

    write_jsonl(prediction_save_path, predictions_noresult)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        default="results/mbpp",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n",
        type=int,
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. ",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--use_markdown",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput.",
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
        "--use_stop_sequences",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
