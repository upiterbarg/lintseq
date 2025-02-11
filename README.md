# Training Language Models on Synthetic Edit Sequences Improves Code Synthesis 

This is the official code release accompanying the paper *[Training Language Models on Synthetic Edit Sequences Improves Code Synthesis](https://arxiv.org/abs/2410.02749)* (ICLR 2025).

---

## Links

- Project page: [https://lintseq.github.io/](https://lintseq.github.io/)
- [TinyCodeLM models are available on HuggingFace](https://huggingface.co/collections/upiter/tinycodelm-6709636f4aba6241d547334f)


---

**Tldr**: LLMs are typically trained to autoregressively synthesize entire programs from scratch. This makes repeatedly editing a program with an LLM extremely expensive. Current state-of-the-art, LLM-powered code editing tools like Cursor [repeatedly prompt models to rewrite entire programs during every edit generation call](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply). We claim that this is the result of a data problem. 

To solve it, we introduce a synthetic data generation algorithm (**LintSeq**). This algorithm refactors programs into sequences of synthetic edits by using a linter to procedurally sample across interdependent lines of source code. Synthetic edits sampled with LintSeq reflect the syntax and semantics of their programming language. 

To test the algorithm, we use it to refactor a dataset of instruction + program pairs into instruction + program-diff-sequence tuples. Then, we fine-tune a series of smaller LMs ranging from 2.6B to 14B parameters on both the re-factored and original versions of this dataset. We perform comprehensive evaluations comparing edit sequence code LMs against baselines on HumanEval, MBPP(+), CodeContests, DS-1000, and BigCodeBench. We show that models fine-tuned to iteratively synthesize code match or outperform baselines on pass@1, and exhibit better scaling across higher pass@k as a function of total test-time FLOPs. 

Finally, we also pretrain our own tiny LMs for code understanding. We show that fine-tuning these models to synthesize code edit-by-edit results in strong performance on HumanEval and MBPP(+) compared to existing code language models of similar scale such as CodeT5+, AlphaCode, and Codex.

---

## Repository Contents

```
> requirements.txt                            # Python env requirements
> src/                                        # Source code
	> data_gen/                                 # Data generation
		> lintseq.py                                # Core implementation of pythonic LintSeq 
		> generate_with_lintseq.py                  # Parallelized generation of synthetic edit sequences with LintSeq + pylint
		> generate_with_random_sampling.py          # (Linter Ablation) Parallelized random sampling of edit sequences
	> utils.py                                  # All key utilities
	> finetune.py                               # Instruction finetune language models with DeepSpeed
	> configs/                                  # Configuration files
		> ds_configs/                               # DeepSpeed configs
	> eval/                                     # Evaluation code
	> scripts/                                  # Launch scripts in bash
		> eval_scripts/                             # Eval scripts for reproducing TinyCodeLM evals locally
		> finetuning_scripts/                       # Example launch scripts for model finetuning experiments
```
---

## Coming soon... 
- Step-by-step instructions/tutorials for implementing all finetuning and pretraining data pre-processing methods described in the paper
