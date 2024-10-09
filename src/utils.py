import contextlib
import io
import numpy as np
import pdb
import random
import tempfile
import torch
import os
from copy import deepcopy
import difflib
from pylint import run_pylint


EOS_TOKEN = "<|endoftext|>"
DIFF_TOKEN = "<|diff|>"
DIFF_DECORATOR_SUBEQ = "@@"

"""This file contains all of the key utility functions for this project."""


def strip_chain_of_thought(response, expected_programming_language="python"):
    """Given an input example "response", strip away any chain-of-thought-like natural language
    by looking for Markdown formatting.

    Return the resultant response.
    """
    suffix = response
    code_chunks = []
    while f"```{expected_programming_language}" in suffix:
        code_chunk_suffix = suffix[
            suffix.find(f"```{expected_programming_language}")
            + len(f"```{expected_programming_language}") :
        ]
        code_chunks += [code_chunk_suffix[: code_chunk_suffix.find("```")]]
        suffix = code_chunk_suffix[code_chunk_suffix.find("```") + len("```") :]

    # merge parsed code chunks into a single string
    if len(code_chunks) > 0:
        out = "\n".join(code_chunks)
        return out.lstrip().rstrip()
    else:
        return response


def inflate_edit_path(code_as_text, edit_sequence):
    """Apply the line indices of edits to the string representing the program/file.

    Return the corresponding code edits, expressed both as sequences of "raw" program
    states and as code "diffs".
    """
    lines = code_as_text.split("\n")

    integrated = []
    total_edit = []

    for step, edit in enumerate(edit_sequence):
        total_edit += edit
        integrated += [deepcopy(total_edit)]

    raw_text_seq = [code_as_text]
    for edit in integrated:
        out = "\n".join([line for (i, line) in enumerate(lines) if not i in edit])
        raw_text_seq += [out]

    del integrated
    del total_edit

    if raw_text_seq[-1] != "":
        raw_text_seq += [""]

    raw_text_seq = raw_text_seq[::-1]

    diff_text_seq = []

    for i in range(len(raw_text_seq) - 1):
        diff_text_seq += [get_diff(raw_text_seq[i], raw_text_seq[i + 1])]
    return raw_text_seq, diff_text_seq


def get_diff(string_one, string_two, n=0):
    """Compute a clean text delta using 'difflib' between two strings.

    Return the text delta as a single string.
    """
    proc_string_one = string_one.strip().splitlines()
    proc_string_two = string_two.strip().splitlines()

    out = "\n".join(
        [
            line
            for i, line in enumerate(
                difflib.unified_diff(
                    proc_string_one,
                    proc_string_two,
                    n=n,
                    fromfile="file1",
                    tofile="file2",
                    lineterm="",
                )
            )
            if i > 1
        ]
    )
    return out


def resolve_edit_path(edit_path, malformed_penalty=0):
    """Resolves the edits in an insertion-only edit sequence."""
    edits = edit_path.rstrip(EOS_TOKEN).split(f"\n{DIFF_TOKEN}")
    diffs = []
    for edit in edits:
        d = edit.count(DIFF_DECORATOR_SUBEQ)
        if d == 0:
            continue
        sub_diffs = [
            e if i == 0 else f"\n{DIFF_DECORATOR_SUBEQ}" + e
            for (i, e) in enumerate(edit.split(f"\n{DIFF_DECORATOR_SUBEQ}"))
        ]
        diffs += sub_diffs

    lines = []

    for diff in diffs:
        if not DIFF_DECORATOR_SUBEQ in diff:
            continue
        try:
            _, context, edit = diff.split(DIFF_DECORATOR_SUBEQ)
        except:
            if malformed_penalty:
                return ""
            else:
                continue
        edit = edit.lstrip()

        try:
            (_, insert_loc) = context.lstrip().rstrip().split(" ")
        except:
            if malformed_penalty:
                return ""
            else:
                continue

        try:
            insert_loc = (
                int(insert_loc.lstrip("+"))
                if not "," in insert_loc
                else int(insert_loc.split(",")[0].lstrip("+"))
            ) - 1
        except:
            if malformed_penalty:
                return ""
            else:
                continue

        edit_lines = [line.lstrip("-").lstrip("+") for line in edit.split("\n")]
        edit_lines = [line for line in edit_lines]

        if insert_loc >= len(lines):
            lines = lines + edit_lines
        else:
            try:
                lines = lines[:insert_loc] + edit_lines + lines[insert_loc:]
            except:
                lines = lines + edit_lines

    code_as_text = "\n".join(lines)
    return code_as_text


def read_contents(file_path):
    with open(file_path, "r") as f:
        contents = "".join(f.readlines())
    return contents


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io(stream):
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream


def file_linter(file, expected_error_traces=None):
    try:
        program_io = io.StringIO()
        with swallow_io(program_io):
            run_pylint(
                (
                    "--disable=C,R",
                    "--reports=n",
                    "--score=n",
                    "--unsafe-load-any-extension=y",
                    "--generated-members=cv2.*",
                    "--msg-template={C}|{msg_id}|{line}|{column}|{msg}",
                    file,
                )
            )
    except BaseException as e:
        pass
    io_stream = program_io.getvalue()
    io_stream = io_stream.split("\n")
    error_traces = []
    for line in io_stream:
        try:
            C, msg_id, line_id, column_id, msg = line.split("|")
        except:
            continue

        # treat indentation warnings as errors
        if C == "W" and msg_id not in ["W0311", "W0231"]:
            continue
        elif msg_id == "E0001" and "on line" in msg:
            line_id = msg[: msg.rfind("(")].rstrip().split(" ")[-1]

        error_trace = (
            msg_id,
            line_id,
            column_id,
            msg,
        )
        if expected_error_traces is None or not error_trace in expected_error_traces:
            error_traces += [error_trace]
    return error_traces


def set_seed_everywhere(seed):
    """Set random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pylint_traces_to_str(error_traces):
    from termcolor import colored

    success = len(error_traces) == 0
    out = (
        colored("code is clean", "green")
        if success
        else colored(
            "errors found: "
            + ", ".join(
                [
                    "{" + f"{code} {line},{column}" + "}"
                    for (code, line, column) in error_traces
                ]
            ),
            "red",
        )
    )
    return out


def dynamic_code_linter(code_as_text, pretty_print=False):
    with tempfile.NamedTemporaryFile(
        delete_on_close=False, suffix=".py", mode="r+"
    ) as fp:
        fp.write(code_as_text)
        fp.seek(0)
        error_traces = file_linter(fp.name)

    if pretty_print:
        out = pylint_traces_to_str(error_traces)
        print(out)

    return error_traces


def apply_mutation(code_as_text, mutation):
    return "\n".join(
        [line for i, line in enumerate(code_as_text.split("\n")) if not i in mutation]
    ), [i for i in range(len((code_as_text.split("\n")))) if not i in mutation]


"""
The remaining utility code is drawn from the Tulu open-source instruction finetuning project by the 
Allen AI Institute.

@article{wang2023far,
  title={How far can camels go? exploring the state of instruction tuning on open resources},
  author={Wang, Yizhong and Ivison, Hamish and Dasigi, Pradeep and Hessel, Jack and Khot, Tushar and 
  Chandu, Khyathi and Wadden, David and MacMillan, Kelsey and Smith, Noah A and Beltagy, Iz and 
  others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={74764--74786},
  year={2023}
}

@article{ivison2023camels,
  title={Camels in a changing climate: Enhancing lm adaptation with tulu 2},
  author={Ivison, Hamish and Wang, Yizhong and Pyatkin, Valentina and Lambert, Nathan and Peters, 
  Matthew and Dasigi, Pradeep and Jang, Joel and Wadden, David and Smith, Noah A and Beltagy, Iz 
  and others},
  journal={arXiv preprint arXiv:2311.10702},
  year={2023}
}

"""


def estimate_pass_at_k(
    num_samples,
    num_correct,
    k,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def load_hf_lm(
    model_name_or_path,
    device_map="auto",
    torch_dtype="auto",
    load_in_8bit=False,
    convert_to_half=False,
    gptq_model=False,
    token=os.getenv("HF_TOKEN", None),
):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        OPTForCausalLM,
        GPTNeoXForCausalLM,
    )

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            device="cuda:0",
            use_triton=True,
            trust_remote_code=trust_remote_code,
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()
    return model


def load_hf_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    use_fast_tokenizer=True,
    padding_side="left",
    token=os.getenv("HF_TOKEN", None),
):
    from transformers import AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token
        )
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just
        # roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    add_special_tokens=True,
    disable_tqdm=False,
    **generation_kwargs,
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                if stop_id_sequences
                else None,
                **generation_kwargs,
            )
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(
                        batch_input_ids.shape[1], batch_outputs.shape[1]
                    ):
                        if any(
                            batch_outputs[
                                output_idx, token_idx : token_idx + len(stop_sequence)
                            ].tolist()
                            == stop_sequence
                            for stop_sequence in stop_id_sequences
                        ):
                            batch_outputs[
                                output_idx, token_idx:
                            ] = tokenizer.pad_token_id
                            break

            batch_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )
            batch_prompts = tokenizer.batch_decode(
                batch_input_ids, skip_special_tokens=True
            )
            batch_prompts = [
                prompt for prompt in batch_prompts for _ in range(num_return_sequences)
            ]
            batch_generations = [
                output[len(prompt) :]
                for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert (
        len(generations) == len(prompts) * num_return_sequences
    ), "number of generations should be equal to number of prompts * num_return_sequences"
    return generations
