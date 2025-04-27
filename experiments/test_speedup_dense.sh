export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=3
dataset=wikitext2


python run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --device cuda:0 \
        --ntokens 64 \
        --batch-size 256 \
        --eval-dataset $dataset 

python run_benchmark.py \
        --model meta-llama/Llama-2-13b-hf \
        --device cuda:0 \
        --ntokens 64 \
        --batch-size 256 \
        --eval-dataset $dataset 