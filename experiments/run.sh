export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=1
dataset=wikitext2

for sparsity in 0.55
do
python run_slicegpt.py \
        --model meta-llama/Llama-2-13b-hf \
        --save-dir /raid0-data/yifan/out_slicegpt/llama-2-13b/$dataset/Llama-2-7b-hf_$sparsity \
        --sparsity $sparsity \
        --device cuda:0 \
        --eval-baseline \
        --cal-batch-size 1 \
        --cal-dataset $dataset \
        --no-wandb
done