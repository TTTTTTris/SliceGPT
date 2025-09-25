export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=2
dataset=wikitext2
mkdir -p ./out_slicegpt/llama-3-8b/$dataset

for sparsity in 0.3
do
python run_slicegpt.py \
        --model meta-llama/Meta-Llama-3-8B \
        --save-dir ./out_slicegpt/llama-3-8b/$dataset/Llama-3-8b-hf_$sparsity \
        --sparsity $sparsity \
        --device cuda:0 \
        --eval-baseline \
        --cal-nsamples 256 \
        --cal-batch-size 1 \
        --cal-dataset $dataset \
        --no-wandb | tee -a log.txt
done

