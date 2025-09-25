export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=3
python run_lm_eval.py \
        --model-path meta-llama/Llama-2-7b-hf \
        --model Llama-2-7b-hf_0.2 \
        --sliced-model-path './' \
        --sparsity 0.2 \
        --tasks piqa \
        --no-wandb