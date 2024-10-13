# Training Language Models on Synthetic Edit Sequences Improves Code Synthesis 

This is the official code release accompanying the paper *[Training Language Models on Synthetic Edit Sequences Improves Code Synthesis](https://arxiv.org/abs/2410.02749)* (arXiv preprint, 2024).


---

**Tldr 💡**: LLMs are typically trained to autoregressively synthesize entire programs from scratch. This makes repeatedly editing a program with an LLM extremely expensive. Current state-of-the-art, LLM-powered code editing tools like Cursor [repeatedly prompt models to rewrite entire programs during every edit generation call](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply). We claim that this is the result of a data problem. 

To solve it, we introduce a synthetic data generation algorithm (**LintSeq**) that can be used to refactor arbitrary code data into code edit sequences. Repeatedly sampling from small LMs (e.g. Phi 3 3.8B) finetuned on synthetic edit sequences yields solutions to HumanEval and MBPP problems that are **competitive with GPT-4 and GPT-4-Omni**, and have total inference costs that are similar to sampling once from the best open-source LLMs (e.g. Llama 3.1 405B).

We also pretrain our own tiny edit sequence code LMs (150M and 400M parameters). **Our models are state-of-the-art in code synthesis on HumanEval and MBPP across pass@k for their size.**

---

## Links

- Project page: [https://lintseq.github.io/](https://lintseq.github.io/)
- TinyCodeLM models [are now available on HuggingFace](https://huggingface.co/collections/upiter/tinycodelm-6709636f4aba6241d547334f)

---

## Navigating the Repo 🗺️

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
	> finetuning_scripts/                       # Example launch scripts for replicating external model finetuning experiments
		> gemma2_2b/                                # Gemma 2, 2B
		> llama3v1_8b/                              # Llama 3.1, 8B
		> phi3_14b/                                 # Phi 3, Medium (14B)
		> phi3_3b/                                  # Phi 3, Mini (3B)
```

---

## Coming soon... ⚙️
- Step-by-step instructions/tutorials for implementing all finetuning and pretraining data pre-processing methods described in the paper
