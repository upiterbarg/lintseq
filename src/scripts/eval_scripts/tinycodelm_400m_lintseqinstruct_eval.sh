export CUDA_VISIBLE_DEVICES=1
export HF_ALLOW_CODE_EVAL=1

MODEL_ALIAS=tinycodelm_400m_lintseq_remote

TEMPERATURE=0.2
TOP_P=0.95
UNBIASED_SAMPLING_SIZE=10
HUMANEVAL_SAVEDIR=results/humaneval/${MODEL_ALIAS}/temp${TEMPERATURE//./_}_topp${TOP_P//./_}_samp${UNBIASED_SAMPLING_SIZE}

wget -nc https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz -P eval_data/
wget -nc https://huggingface.co/datasets/bigcode/humanevalpack/raw/main/data/python/data/humanevalpack.jsonl -P eval_data/

python src/eval/humaneval/run_eval.py \
    --data_file eval_data/HumanEval.jsonl.gz \
    --unbiased_sampling_size_n $UNBIASED_SAMPLING_SIZE \
    --temperature $TEMPERATURE \
    --save_dir $HUMANEVAL_SAVEDIR \
    --model upiter/TinyCodeLM-400M-LintSeqInstruct \
    --tokenizer upiter/TinyCodeLM-400M-LintSeqInstruct \
    --use_vllm \
    --diff 1 \
    --prompt_version 1 \
    --top_p 0.95 \

python src/eval/humaneval/run_exec.py \
    --data_file eval_data/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 \
    --save_dir $HUMANEVAL_SAVEDIR \
    --diff 1


TEMPERATURE=0.0
TOP_P=0.95
UNBIASED_SAMPLING_SIZE=1
MBPP_SAVEDIR=results/mbpp/${MODEL_ALIAS}/temp${TEMPERATURE//./_}_topp${TOP_P//./_}_samp${UNBIASED_SAMPLING_SIZE}


python src/eval/mbpp/run_eval.py \
    --unbiased_sampling_size_n $UNBIASED_SAMPLING_SIZE \
    --temperature $TEMPERATURE \
    --save_dir $MBPP_SAVEDIR \
    --model upiter/TinyCodeLM-400M-LintSeqInstruct  \
    --tokenizer upiter/TinyCodeLM-400M-LintSeqInstruct \
    --use_vllm \
    --diff 1 \
    --prompt_version 1 \

python src/eval/mbpp/run_exec.py \
    --eval_pass_at_ks 1 \
    --save_dir $MBPP_SAVEDIR \
    --diff 1
